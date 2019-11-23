from toolbox.ui import UI
from encoder import inference as encoder
from synthesizer.inference import Synthesizer
from vocoder import inference as vocoder
from pathlib import Path
from time import perf_counter as timer
from toolbox.utterance import Utterance
import numpy as np
import traceback
import sys
import re
import time
from synthesizer import audio, hparams
from tools import find_start_end_points
from .sentence import xinqing_texts

# Use this directory structure for your datasets, or modify it to fit your needs
recognized_datasets = [
    r'stcmds\stcmds\wavs',
    r"aishell\aishell\wavs",
    r"xunfei",
    r"biaobei",
    r"fansen\audio",
    r"record",
    r"librispeech\LibriSpeech\test-clean\LibriSpeech\test-clean",
]

_out_dir = Path('toolbox/saved_files')
_out_dir.mkdir(exist_ok=True)

_out_mel_dir = _out_dir.joinpath('mels')
_out_mel_dir.mkdir(exist_ok=True)

_out_wav_dir = _out_dir.joinpath('wavs')
_out_wav_dir.mkdir(exist_ok=True)

_out_embed_dir = _out_dir.joinpath('embeds')
_out_embed_dir.mkdir(exist_ok=True)

_out_record_dir = _out_dir.joinpath('records')
_out_record_dir.mkdir(exist_ok=True)

filename_formatter_re = re.compile(r'[\s\\/:*?"<>|\']+')
filename_formatter = lambda x: filename_formatter_re.sub('_', x)[:100]

eval_texts = """我家朵朵是世界上最漂亮的朵朵。
知道自己是什么样的人
要做什么
无需活在别人非议或期待里
你勤奋充电努力工作
对人微笑
是为了扮靓自己照亮自己的心
告诉自己
我是一股独立向上的力量
一个人的自愈的能力越强
才越有可能接近幸福""".split("\n")

total_texts = xinqing_texts


class Toolbox:
    def __init__(self, datasets_root, enc_models_dir, syn_models_dir, voc_models_dir, low_mem):
        sys.excepthook = self.excepthook
        self.datasets_root = datasets_root
        self.low_mem = low_mem
        self.utterances = set()
        self.current_generated = (None, None, None, None)  # speaker_name, spec, breaks, wav

        self.synthesizer = None  # type: Synthesizer

        # Initialize the events and the interface
        self.ui = UI()
        self.reset_ui(enc_models_dir, syn_models_dir, voc_models_dir)
        self.setup_events()
        self.ui.start()

    def excepthook(self, exc_type, exc_value, exc_tb):
        traceback.print_exception(exc_type, exc_value, exc_tb)
        self.ui.log("Exception: %s" % exc_value)

    def setup_events(self):
        # Dataset, speaker and utterance selection
        self.ui.browser_load_button.clicked.connect(lambda: self.load_from_browser())
        random_func = lambda level: lambda: self.ui.populate_browser(self.datasets_root,
                                                                     recognized_datasets,
                                                                     level)
        text_func = lambda: self.ui.text_prompt.setPlainText(np.random.choice(total_texts, 1)[0])
        self.ui.random_dataset_button.clicked.connect(text_func)
        self.ui.random_speaker_button.clicked.connect(random_func(1))
        self.ui.random_utterance_button.clicked.connect(random_func(2))
        self.ui.dataset_box.currentIndexChanged.connect(random_func(1))
        self.ui.speaker_box.currentIndexChanged.connect(random_func(2))

        # Model selection
        self.ui.encoder_box.currentIndexChanged.connect(self.init_encoder)

        def func():
            self.synthesizer = None

        self.ui.synthesizer_box.currentIndexChanged.connect(func)
        self.ui.vocoder_box.currentIndexChanged.connect(self.init_vocoder)

        # Utterance selection
        func = lambda: self.load_from_browser(self.ui.browse_file())
        self.ui.browser_browse_button.clicked.connect(func)
        func = lambda: self.ui.draw_utterance(self.ui.selected_utterance, "current")
        self.ui.utterance_history.currentIndexChanged.connect(func)
        func = lambda: self.ui.play(self.ui.selected_utterance.wav, Synthesizer.sample_rate)
        self.ui.play_button.clicked.connect(func)
        self.ui.stop_button.clicked.connect(self.ui.stop)
        self.ui.record_button.clicked.connect(self.record)

        # Generation
        func = lambda: self.synthesize() or self.vocode()
        self.ui.generate_button.clicked.connect(func)
        self.ui.synthesize_button.clicked.connect(self.synthesize)
        self.ui.vocode_button.clicked.connect(self.vocode)

        # UMAP legend
        self.ui.clear_button.clicked.connect(self.clear_utterances)

    def reset_ui(self, encoder_models_dir, synthesizer_models_dir, vocoder_models_dir):
        self.ui.populate_browser(self.datasets_root, recognized_datasets, 0, True)
        self.ui.populate_models(encoder_models_dir, synthesizer_models_dir, vocoder_models_dir)

    def load_from_browser(self, fpath=None):
        if fpath is None:
            fpath = Path(self.datasets_root,
                         self.ui.current_dataset_name,
                         self.ui.current_speaker_name,
                         self.ui.current_utterance_name)
            # name = '-'.join(fpath.relative_to(self.datasets_root).parts)
            speaker_name = "-".join((self.ui.current_dataset_name.replace("\\", "_").replace("/", "_"),
                                     self.ui.current_speaker_name.replace("\\", "_").replace("/", "_")))
            name = "-".join((speaker_name, self.ui.current_utterance_name.replace("\\", "_").replace("/", "_")))
            # name = '-'.join(fpath.relative_to(self.datasets_root.joinpath(self.ui.current_dataset_name)).parts)
            # speaker_name = self.ui.current_speaker_name.replace("\\", "-").replace("/", "-")
            # Select the next utterance
            if self.ui.auto_next_checkbox.isChecked():
                self.ui.browser_select_next()
        elif fpath == "":
            return
        else:
            name = fpath.name
            speaker_name = fpath.parent.name

        # Get the wav from the disk. We take the wav with the vocoder/synthesizer format for
        # playback, so as to have a fair comparison with the generated audio
        wav = Synthesizer.load_preprocess_wav(fpath)
        self.ui.log("Loaded %s" % name)

        self.add_real_utterance(wav, name, speaker_name)

    def record(self):
        wav = self.ui.record_one(encoder.sampling_rate, 5)
        if wav is None:
            return

        self.ui.play(wav, encoder.sampling_rate)

        speaker_name = "user01"
        name = speaker_name + "_rec_%d" % int(time.time())
        audio.save_wav(wav, _out_record_dir.joinpath(name + '.wav'), encoder.sampling_rate)  # save

        self.add_real_utterance(wav, name, speaker_name)

    def add_real_utterance(self, wav, name, speaker_name):
        # Compute the mel spectrogram
        spec = Synthesizer.make_spectrogram(wav)
        self.ui.draw_spec(spec, "current")

        # Compute the embedding
        if not encoder.is_loaded():
            self.init_encoder()
        encoder_wav = encoder.preprocess_wav(wav)
        embed, partial_embeds, _ = encoder.embed_utterance(encoder_wav, return_partials=True)

        np.save(_out_embed_dir.joinpath(name + '.npy'), embed, allow_pickle=False)  # save

        # Add the utterance
        utterance = Utterance(name, speaker_name, wav, spec, embed, partial_embeds, False)
        self.utterances.add(utterance)
        self.ui.register_utterance(utterance)

        # Plot it
        self.ui.draw_embed(embed, name, "current")
        self.ui.draw_umap_projections(self.utterances)

    def clear_utterances(self):
        self.utterances.clear()
        self.ui.draw_umap_projections(self.utterances)

    def synthesize(self):
        self.ui.log("Generating the mel spectrogram...")
        self.ui.set_loading(1)

        # Synthesize the spectrogram
        if self.synthesizer is None:
            model_dir = self.ui.current_synthesizer_model_dir
            checkpoints_dir = model_dir.joinpath("taco_pretrained")
            self.synthesizer = Synthesizer(checkpoints_dir, low_mem=self.low_mem)
        if not self.synthesizer.is_loaded():
            self.ui.log("Loading the synthesizer %s" % self.synthesizer.checkpoint_fpath)

        texts = self.ui.text_prompt.toPlainText().split("\n")
        embed = self.ui.selected_utterance.embed
        embeds = np.stack([embed] * len(texts))
        specs = self.synthesizer.synthesize_spectrograms(texts, embeds)

        # 去除前后安静或噪声部分
        for num, spec in enumerate(specs):
            tmp = spec.T
            sidx, eidx = find_start_end_points(tmp)
            specs[num] = tmp[sidx:eidx].T

        # specs = [spec.T[:find_endpoint(spec.T)].T for spec in specs]  # find endpoint
        breaks = [spec.shape[1] for spec in specs]
        spec = np.concatenate(specs, axis=1)

        fref = '-'.join([self.ui.current_dataset_name, self.ui.current_speaker_name, self.ui.current_utterance_name])
        ftext = '。'.join(texts)
        ftime = '{}'.format(int(time.time()))
        fname = filename_formatter('{}_{}_{}zi_{}.npy'.format(fref, ftime, len(ftext), ftext))
        np.save(_out_mel_dir.joinpath(fname), spec, allow_pickle=False)  # save

        self.ui.draw_spec(spec, "generated")
        self.current_generated = (self.ui.selected_utterance.speaker_name, spec, breaks, None)
        self.ui.set_loading(0)

    def vocode(self):
        speaker_name, spec, breaks, _ = self.current_generated
        assert spec is not None

        # Synthesize the waveform
        if not vocoder.is_loaded():
            self.init_vocoder()

        def vocoder_progress(i, seq_len, b_size, gen_rate):
            real_time_factor = (gen_rate / Synthesizer.sample_rate) * 1000
            line = "Waveform generation: %d/%d (batch size: %d, rate: %.1fkHz - %.2fx real time)" \
                   % (i * b_size, seq_len * b_size, b_size, gen_rate, real_time_factor)
            self.ui.log(line, "overwrite")
            self.ui.set_loading(i, seq_len)

        if self.ui.current_vocoder_fpath is not None:
            self.ui.log("")
            wav = vocoder.infer_waveform(spec, progress_callback=vocoder_progress)
        else:
            self.ui.log("Waveform generation with Griffin-Lim... ")
            wav = Synthesizer.griffin_lim(spec)
        self.ui.set_loading(0)
        self.ui.log(" Done!", "append")

        # Add breaks
        b_ends = np.cumsum(np.array(breaks) * Synthesizer.hparams.hop_size)
        b_starts = np.concatenate(([0], b_ends[:-1]))
        wavs = [wav[start:end] for start, end, in zip(b_starts, b_ends)]
        breaks = [np.zeros(int(0.15 * Synthesizer.sample_rate))] * len(breaks)
        wav = np.concatenate([i for w, b in zip(wavs, breaks) for i in (w, b)])

        # Play it
        wav = wav / np.abs(wav).max() * 0.97
        self.ui.play(wav, Synthesizer.sample_rate)

        fref = '-'.join([self.ui.current_dataset_name, self.ui.current_speaker_name, self.ui.current_utterance_name])
        ftime = '{}'.format(int(time.time()))
        ftext = self.ui.text_prompt.toPlainText()
        fms = int(len(wav) * 1000 / Synthesizer.sample_rate)
        fname = filename_formatter('{}_{}_{}ms_{}.wav'.format(fref, ftime, fms, ftext))
        audio.save_wav(wav, _out_wav_dir.joinpath(fname), Synthesizer.sample_rate)  # save

        # Compute the embedding
        # TODO: this is problematic with different sampling rates, gotta fix it
        if not encoder.is_loaded():
            self.init_encoder()
        encoder_wav = encoder.preprocess_wav(wav)
        embed, partial_embeds, _ = encoder.embed_utterance(encoder_wav, return_partials=True)

        # Add the utterance
        name = speaker_name + "_gen_%05d" % int(time.time())
        utterance = Utterance(name, speaker_name, wav, spec, embed, partial_embeds, True)

        np.save(_out_embed_dir.joinpath(name + '.npy'), embed, allow_pickle=False)  # save

        self.utterances.add(utterance)

        # Plot it
        self.ui.draw_embed(embed, name, "generated")
        self.ui.draw_umap_projections(self.utterances)

    def init_encoder(self):
        model_fpath = self.ui.current_encoder_fpath

        self.ui.log("Loading the encoder %s... " % model_fpath)
        self.ui.set_loading(1)
        start = timer()
        encoder.load_model(model_fpath)
        self.ui.log("Done (%dms)." % int(1000 * (timer() - start)), "append")
        self.ui.set_loading(0)

    def init_vocoder(self):
        model_fpath = self.ui.current_vocoder_fpath
        # Case of Griffin-lim
        if model_fpath is None:
            return

        self.ui.log("Loading the vocoder %s... " % model_fpath)
        self.ui.set_loading(1)
        start = timer()
        vocoder.load_model(model_fpath)
        self.ui.log("Done (%dms)." % int(1000 * (timer() - start)), "append")
        self.ui.set_loading(0)
