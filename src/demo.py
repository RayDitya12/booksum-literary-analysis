import argparse
import os

import torch

import config

from excerpt import select_representative
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class Analyzer:
    def __init__(self, model_dir):
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.arm = os.path.basename(model_dir)
        cfg = config.ARMS.get(self.arm, {})
        self.max_src = cfg.get("max_src_tokens", 1024)
        self.hybrid_input = cfg.get("src_field") == "chapter_selected"

    @torch.no_grad()
    def run(self, chapter, use_selection=None, num_beams=1, max_new=512):
        if use_selection is None:
            use_selection = self.hybrid_input
        text = (
            select_representative(chapter, max_words=config.HYBRID_MAX_WORDS, buckets=config.HYBRID_SENTENCE_BUCKETS)
            if use_selection
            else chapter
        )
        enc = self.tok([str(text)], max_length=self.max_src, truncation=True, padding=True, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **enc,
            max_new_tokens=max_new,
            num_beams=max(1, num_beams),
            early_stopping=num_beams > 1,
            do_sample=False,
        )
        return self.tok.batch_decode(out, skip_special_tokens=True)[0]


def build(app_name="m1_bart_direct"):
    import gradio as gr

    model_dir = os.path.join(config.OUT_DIR, app_name)
    if not os.path.isdir(model_dir):
        raise SystemExit(f"model folder not found: {model_dir}. Train it first, or pass --model <dir>.")
    a = Analyzer(model_dir)

    def respond(chapter, use_selection, beams):
        if not chapter or not chapter.strip():
            return "Type or paste a chapter first."
        return a.run(chapter, use_selection=use_selection, num_beams=int(beams), max_new=512)

    with gr.Blocks(title="BookSum Literary Analysis Generator") as demo:
        gr.Markdown("# Chapter → Literary Analysis (BookSum — kelompok 2A INT)")
        gr.Markdown(f"Model: `{a.arm}` · input window `{a.max_src}` token · arsitektur `{config.ARMS.get(a.arm, {}).get('model', 'custom')}`")
        with gr.Row():
            with gr.Column():
                inp = gr.Textbox(label="Chapter text", lines=14, placeholder="Paste a novel chapter here")
                sel = gr.Checkbox(label="Compress with TF-IDF selection first (needed for 1024-token models)", value=a.hybrid_input)
                beams = gr.Slider(1, 6, value=1, step=1, label="Beam size")
                run = gr.Button("Generate analysis", variant="primary")
            out = gr.Textbox(label="Generated literary analysis", lines=18)
        run.click(respond, [inp, sel, beams], [out])
        gr.Examples(
            examples=[["It was a feature peculiar to the colonial wars of North America, that the toils and dangers of the wilderness were to be encountered before the adverse hosts could meet."]],
            inputs=[inp],
        )
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="path to trained arm folder")
    ap.add_argument("--arm", default="m1_bart_direct")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    path = args.model or os.path.join(config.OUT_DIR, args.arm)
    build(os.path.basename(path)).launch(server_port=args.port, share=torch.cuda.is_available())
