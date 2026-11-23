import gradio as gr
from config import DEMO_DIR, EXAMPLE
from video import process_video


def run(path, use_tracker, seconds):
	try:
		return process_video(path, use_tracker, seconds)
	except ValueError as e:
		raise gr.Error(str(e))


with gr.Blocks(title='Lane tracking') as app:
	gr.Markdown('Blue: left line. Orange: right line. Yellow: predicted by the tracker. Green: your lane.')
	with gr.Row():
		with gr.Column():
			video = gr.Video(label='Windshield video')
			use_tracker = gr.Checkbox(value=True, label='Kalman tracker')
			seconds = gr.Slider(1, 120, value=20, step=1, label='Max seconds')
			button = gr.Button('Run', variant='primary')
			gr.Examples([[EXAMPLE]], inputs=[video])
		with gr.Column():
			outputs = [gr.Video(label='Result'), gr.Dataframe(label='Summary'), gr.File(label='Per-frame CSV')]
	button.click(run, [video, use_tracker, seconds], outputs)

app.launch(inbrowser=True, allowed_paths=[DEMO_DIR])