"""Export one-frame LS-EEND streaming step with explicit recurrent state.

The exported graph is deliberately one-step and fixed-shape. State tensors are
inputs/outputs so the host/board loop can preserve native LS-EEND retention and
convolution state across the complete recording.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import torch
import torch.nn as nn

from paths import LS, CKPT, CONFIG_YAML, EXPORT_DIR, add_upstream_to_syspath
add_upstream_to_syspath()
from nnet.model.onl_conformer_retention_enc_1dcnn_tfm_retention_enc_linear_non_autoreg_pos_enc_l2norm_emb_loss_mask import OnlineConformerRetentionDADiarization
from nnet.modules.retention import MultiScaleRetention
import hyperpyyaml


def _mean_form_recurrent_forward(self, qr, kr, v, decay, incremental_state):
    """Bounded-state recurrence, numerically equivalent to the upstream one.

    Upstream keeps ``kv_t = K_t / sqrt(t)`` where ``K_t = sum_i k_i v_i`` and
    ``scale_t = t`` (this checkpoint has ``decay == 1`` for every head, see
    ``RetNetRelPos``: ``decay = log([1,1,1,1])``). ``kv_t`` therefore grows like
    ``sqrt(t)`` -- reaching +-45k over a 1921-frame recording -- which makes a
    single U16 quantization scale useless for the early frames.

    Here the state is the running mean ``M_t = K_t / t = kv_t / sqrt(t)``, which
    is bounded. The retention output only feeds ``group_norm`` (LayerNorm with
    ``elementwise_affine=False``), which is invariant to a positive scalar
    factor, so dropping the ``sqrt(t)`` gain leaves the layer output unchanged
    up to the LayerNorm epsilon.

    The ``1/t`` mixing coefficient is supplied by the caller through
    ``incremental_state['inv_count']`` instead of being tracked as an unbounded
    counter state.

    The raw per-frame term ``k_t v_t`` is also published as ``increment`` so the
    caller can carry the accumulator itself in FP32. Feeding a host-accumulated
    state back means quantization error stays per-frame instead of compounding
    through the recurrence, and it avoids the quantized ``(kv-mean)*1/t`` update
    rounding to zero once ``t`` is large.
    """
    bsz = v.size(0)
    v = v.view(bsz, self.num_heads, self.head_dim, 1)
    kv = kr * v
    b = incremental_state["inv_count"]  # (1, num_heads, 1, 1), equals 1/t
    prev_mean = incremental_state["prev_key_value"]
    mean = prev_mean + (kv - prev_mean) * b
    incremental_state["increment"] = kv
    incremental_state["prev_key_value"] = mean
    return torch.sum(qr * mean, dim=3)


# Apply only in this export process; the upstream source repository is untouched.
MultiScaleRetention.recurrent_forward = _mean_form_recurrent_forward

class StreamingStep(nn.Module):
    def __init__(self, model, max_nspks=10):
        super().__init__(); self.model=model; self.max_nspks=max_nspks
        self.nenc=len(model.enc.encoder.layers); self.ndec=len(model.dec.layers)
        self.h=model.n_units; self.heads=model.enc.encoder.layers[0].sequential[1].module.self_attn.num_heads
        self.keydim=self.h//self.heads; self.valdim=self.h//self.heads
        self.convk=model.enc.encoder._conv_kernel_size; self.outk=2*model.delay+1
        self.register_buffer('decay', torch.ones(self.heads))

    def forward(self, feat, inv_count, dec_inv_count, conv_cache, *states):
        # state order: enc_mean, enc_conv per encoder layer, then decoder mean.
        # `inv_count` / `dec_inv_count` are 1/t for the encoder and decoder streams;
        # the caller supplies them so no unbounded counter state is needed.
        idx=0; enc_ret=[]; enc_conv=[]
        for _ in range(self.nenc):
            enc_ret.append({'prev_key_value':states[idx], 'inv_count':inv_count}); idx+=1
            enc_conv.append(states[idx]); idx+=1
        dec_ret=[]
        for _ in range(self.ndec):
            dec_ret.append({'prev_key_value':states[idx], 'inv_count':dec_inv_count}); idx+=1

        # RetNetRelPos in this checkpoint has decay=1 for every head and the
        # sin/cos terms are not used by MultiScaleRetention (theta_shift is
        # disabled), so the recurrent transition is position independent.
        emb=self.model.enc.forward_one_step(feat, 0, enc_ret, enc_conv)
        # Causal streaming CNN, explicit cache (B,D,K-1).
        x=emb.transpose(1,2)
        win=torch.cat([conv_cache,x],dim=2)
        conv=torch.nn.functional.conv1d(win,self.model.cnn.weight,self.model.cnn.bias)
        new_conv_cache=win[:,:,1:]
        conv=conv.transpose(1,2)
        conv=conv/(torch.norm(conv,dim=-1,keepdim=True)+1e-6)
        # The native loop suppresses the decoder entirely until StreamingConv1d
        # starts emitting (call `delay+1`). That gating is done by the caller: during
        # warmup it discards `pred` and re-feeds the previous decoder states instead
        # of these outputs, so no branch is needed inside the graph.
        attr=self.model.dec.forward_one_step(conv, 0, self.max_nspks, dec_ret)
        attr=attr/(torch.norm(attr,dim=-1,keepdim=True)+1e-6)
        y=torch.matmul(conv.unsqueeze(-2),attr.transpose(-1,-2)).squeeze(-2)
        outputs=[y]
        # Emit the bounded per-frame increment k_t*v_t, not the updated mean: the
        # caller accumulates in FP32 so quantization error cannot compound.
        for i, s in enumerate(enc_ret): outputs += [s['increment'],enc_conv[i]]
        outputs += [new_conv_cache]
        for s in dec_ret: outputs += [s['increment']]
        return tuple(outputs)

def load():
    cfgp=CONFIG_YAML
    with open(cfgp) as f: cfg=hyperpyyaml.load_hyperpyyaml(f)
    D=(2*cfg['data']['context_recp']+1)*cfg['data']['feat']['n_mels']
    m=OnlineConformerRetentionDADiarization(n_speakers=cfg['data']['num_speakers'],in_size=D,**cfg['model']['params'])
    sd=torch.load(CKPT,map_location='cpu',weights_only=False)
    if isinstance(sd,dict) and 'state_dict' in sd: sd=sd['state_dict']
    sd={(k[6:] if k.startswith('model.') else k):v for k,v in sd.items()}
    sd={k.replace('dec.attractor_decoder.layers.','dec.layers.'):v for k,v in sd.items()}
    m.load_state_dict(sd,strict=False); return m.eval(),cfg

def main():
    EXPORT_DIR.mkdir(parents=True,exist_ok=True)
    m,cfg=load()
    # Each release is trained with a different max_speakers, so the number of
    # output channels (and the decoder state batch) follows the infer YAML:
    # simu=8->10, AMI=4->6, CALLHOME=7->9, DIHARD2/3=10->12.
    max_nspks=cfg['data']['max_speakers']+2
    wrap=StreamingStep(m,max_nspks).eval()
    B=1; D=345; H=256; heads=4; K=16; DK=18
    print(f'config {CONFIG_YAML.name}: max_speakers={cfg["data"]["max_speakers"]} -> {max_nspks} channels')
    feat=torch.randn(B,1,D)
    conv_cache=torch.zeros(B,H,DK)
    inv_count=torch.ones(1,heads,1,1); dec_inv_count=torch.ones(1,heads,1,1)
    names=['feat','inv_count','dec_inv_count','conv_cache']
    inputs=[feat,inv_count,dec_inv_count,conv_cache]
    for i in range(4):
        names += [f'enc{i}_kv',f'enc{i}_conv']
        inputs += [torch.zeros(B,heads,H//heads,H//heads),torch.zeros(B,H,K-1)]
    for i in range(2):
        names += [f'dec{i}_kv']
        inputs += [torch.zeros(B*max_nspks,heads,H//heads,H//heads)]
    outputs = ['pred']
    for i in range(4): outputs += [f'enc{i}_inc', f'enc{i}_conv_out']
    outputs += ['conv_cache_out']
    for i in range(2): outputs += [f'dec{i}_inc']
    with torch.no_grad():
        torch_outputs=wrap(*inputs)
        ref=torch_outputs[0]
    onnx_path=EXPORT_DIR/'streaming_step.onnx'
    torch.onnx.export(wrap,tuple(inputs),str(onnx_path),input_names=names,output_names=outputs,opset_version=17,dynamo=False,do_constant_folding=True)
    meta={'model_name':'ls_eend_streaming_step','config':CONFIG_YAML.name,'max_speakers':cfg['data']['max_speakers'],'max_nspks':max_nspks,'task':'streaming_speaker_diarization','input_names':names,'output_names':outputs,'input_shapes':{n:list(x.shape) for n,x in zip(names,inputs)},'notes':'frame-wise state model in bounded mean form. Retention states hold the running mean K_t/t instead of the upstream K_t/sqrt(t), so magnitudes stay bounded and a single U16 scale covers the whole recording; the dropped sqrt(t) gain is cancelled exactly by the following affine-free LayerNorm. The graph emits the bounded per-frame increment enc{i}_inc/dec{i}_inc, and the CALLER accumulates mean += (inc-mean)/t in FP32 so quantization error stays per-frame. Caller also feeds inv_count=1/t (encoder) and dec_inv_count=1/t_dec (decoder), and MUST hold decoder state frozen for the first 9 frames (conv warmup) while discarding those preds, matching native StreamingConv1d.'}
    (EXPORT_DIR/'model_meta.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    torch.save(ref,EXPORT_DIR/'sample_output.pt')
    print('exported',onnx_path)
    print('inputs',len(names),'outputs',len(outputs),'ref',tuple(ref.shape))
    import onnx
    import onnxruntime as ort
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    sess=ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    feed={n:x.detach().cpu().numpy() for n,x in zip(names,inputs)}
    ort_outputs=sess.run(outputs, feed)
    cosines=[]; max_diffs=[]
    for torch_out, ort_out in zip(torch_outputs, ort_outputs):
        a=torch_out.detach().cpu().numpy().astype('float32').reshape(-1)
        b=ort_out.astype('float32').reshape(-1)
        cosines.append(float((a@b)/(max((a@a)**0.5*(b@b)**0.5,1e-12))))
        max_diffs.append(float(abs(a-b).max()))
    meta['torch_onnx_one_step_min_cosine']=min(cosines)
    meta['torch_onnx_one_step_max_diff']=max(max_diffs)
    meta['torch_onnx_one_step_output_cosines']=cosines
    (EXPORT_DIR/'model_meta.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('torch-onnx one-step min cosine',min(cosines),'max diff',max(max_diffs))
if __name__=='__main__': main()
