import os
import shutil
import collections as clt
from pathlib import Path
from tqdm import tqdm
from pypinyin.style import convert
from pypinyin.constants import PHRASES_DICT, PINYIN_DICT
from pypinyin import lazy_pinyin
from unidecode import unidecode
import re
import json

_discard_pattern = re.compile(r'[a-zA-Z0-9]')


def is_discard(text):
    return True if _discard_pattern.match(text) else False


def get_pinyin(text):
    pnys = lazy_pinyin(text, style=8, errors=lambda x: list(unidecode(x)))
    return ' '.join(pnys)


def run_stcmds2librispeechformat():
    in_dir = r'E:\data\stcmds\ST-CMDS-20170001_1-OS'
    out_dir = r'E:\data\stcmds\wavs'
    speaker_pathlst_dt = clt.defaultdict(list)
    for fname in sorted(os.listdir(in_dir)):
        speaker = fname[8:15]
        speaker_pathlst_dt[speaker].append(fname)

    for speaker, pathlst in tqdm(speaker_pathlst_dt.items()):
        speaker_dir = os.path.join(out_dir, speaker)
        os.makedirs(speaker_dir, exist_ok=True)
        for num, fname in enumerate(pathlst):
            book = "{:03d}".format(num // 300 + 1)
            inpath = os.path.join(in_dir, fname)
            book_dir = os.path.join(speaker_dir, book)
            os.makedirs(book_dir, exist_ok=True)
            outpath = os.path.join(book_dir, fname)
            shutil.copy(inpath, outpath)


def run_create_trans():
    in_dir = Path('E:\data\stcmds\wavs')
    for txt_path in tqdm(sorted(in_dir.glob('*/*/2017*.txt'))):
        book_dir = txt_path.parent
        *_, speaker, book = book_dir.parts
        trans_path = book_dir / '{}-{}.trans.txt'.format(speaker, book)
        with txt_path.open('r', encoding='utf8') as fin, trans_path.open('a', encoding='utf8') as fout:
            text = fin.readlines()[0]
            pny = get_pinyin(text.strip())
            name = txt_path.stem
            out = '{}\t{}\n'.format(name, pny)
            if is_discard(text):
                print(text)
            fout.write(out)


def run_create_json():
    in_dir = Path('E:\data\stcmds\wavs')
    outpath = in_dir.parent / 'meta_json.txt'
    fout = outpath.open('w', encoding='utf8')
    for trans_path in tqdm(sorted(in_dir.glob('*/*/*.trans.txt'))):
        book_dir = trans_path.parent
        *_, speaker, book = book_dir.parts
        with trans_path.open('r', encoding='utf8') as fin:
            for line in fin:
                name, pny = line.strip().split('\t')
                text = book_dir.joinpath(name + '.txt').open('r', encoding='utf8').readlines()[0].strip()
                dt = {'index': name, 'text': text, 'pinyin': pny,
                      'path': '{}/{}/{}/{}.wav'.format('wavs', speaker, book, name)}
                out = json.dumps(dt, ensure_ascii=False)
                fout.write('{}\n'.format(out))
    fout.close()


def run_audio_strip():
    from pydub import AudioSegment
    from pydub.silence import split_on_silence, detect_silence
    from pathlib import Path
    indir = Path(r'E:\lab\zhrtvc\zhrtvc\toolbox\saved_files\records')
    for num, inpath in enumerate(indir.glob('*')):
        aud = AudioSegment.from_wav(inpath)
        outs = detect_silence(aud, min_silence_len=200, silence_thresh=-48)
        print(outs)
        if num == 7:
            aud[outs[0][-1] - 100: outs[-2][0] + 200].export(indir.parent.joinpath('kdd-' + inpath.name[-8:]),
                                                             format='wav')


def run_abspath_train_txt():
    inpath = Path(r'F:\data\aishell\SV2TTS\synthesizer\train.txt')
    outpath = Path(r'E:\data\train.txt')
    with inpath.open('r', encoding='utf8') as fin, outpath.open('a', encoding='utf8') as fout:
        for line in fin:
            # audio-000001_00.npy|mel-000001_00.npy|embed-000001_00.npy|42560|213|ka3 er3 pu3 pei2 wai4 sun1 wan2 hua2 ti1 .
            parts = line.strip('\n').split('|')
            parts[0] = str(inpath.parent.joinpath('audio', parts[0])).replace('\\', '/')
            parts[1] = str(inpath.parent.joinpath('mels', parts[1])).replace('\\', '/')
            parts[2] = str(inpath.parent.joinpath('embeds', parts[2])).replace('\\', '/')
            out = '|'.join(parts)
            fout.write('{}\n'.format(out))


def run_embed_sim():
    import shutil
    from tools.enc_analyzer import get_refs, get_speakers
    import collections as clt
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
    from matplotlib import pyplot as plt
    indir = Path(r'E:\data\stcmds\SV2TTS\synthesizer')
    inpath = indir.joinpath('train.txt')
    speaker_embeds_dict = clt.defaultdict(list)
    for line in inpath.open('r', encoding='utf8'):
        fname = line.split('|')[2]
        # embed-20170001P00001A0001_00.npy
        speaker = fname[14:21]
        speaker_embeds_dict[speaker].append(indir.joinpath('embeds', fname))

    data_lst = []
    names = []
    for num, (speaker, embeds) in enumerate(tqdm(speaker_embeds_dict.items())):
        if num >= 1000:
            break
        fpaths = embeds  # np.random.choice(embeds, 1, replace=False)
        sp_datas = []
        for fpath in fpaths:
            data = np.load(fpath)
            sp_datas.append(data)
        sp_datas = np.array(sp_datas)
        outs = get_refs(sp_datas, num=1)
        data_lst.append(sp_datas[outs[0]])
        names.append(embeds[outs[0]].name[6:-7])

    wav_dir = Path(r'E:\data\stcmds\ST-CMDS-20170001_1-OS')
    out_dir = wav_dir.parent.joinpath('refs')
    out_dir.mkdir(exist_ok=True)
    for name in tqdm(names):
        shutil.copy(wav_dir.joinpath(name + '.wav'), out_dir.joinpath(name + '.wav'))

    data_lst = np.array(data_lst)

    ref_lst = get_refs(data_lst, 10)
    sp_lst = get_speakers(data_lst, 100)
    # sp_lst = np.random.choice(range(len(data_lst)), 20, replace=False)
    out = pairwise_distances(data_lst[sp_lst], metric='cosine')
    print(len(out[out < 0.4]), np.mean(out))
    plt.imshow(out, 'jet')
    plt.colorbar()
    plt.show()


def run_get_speakers():
    import shutil
    import numpy as np
    from tools.enc_analyzer import get_speakers
    embed_dir = Path(r'E:\data\stcmds\SV2TTS\synthesizer\embeds')
    ref_dir = Path(r'E:\data\stcmds\refs')
    fpaths = list(ref_dir.glob('*'))
    data_lst = []
    for fpath in tqdm(fpaths):
        # embed-20170001P00001A0001_00.npy
        fname = f'embed-{fpath.stem}_00.npy'
        embed_path = embed_dir.joinpath(fname)
        data_lst.append(np.load(embed_path))

    data_lst = np.array(data_lst)

    sp_lst = get_speakers(data_lst, 100)
    out_dir = Path(r'E:\data\stcmds\speakers100')
    out_dir.mkdir(exist_ok=True)
    for idx in sp_lst:
        shutil.copy(fpaths[idx], out_dir.joinpath(fpaths[idx].name))


def run_mel_strip():
    import numpy as np
    from tools.spec_processor import find_endpoint, find_silences
    from synthesizer.audio import inv_mel_spectrogram, save_wav
    from synthesizer.hparams import hparams
    from matplotlib import pyplot as plt
    inpath = Path(
        r'E:\lab\zhrtvc\zhrtvc\toolbox\saved_files\mels\wavs-P00173I-001_20170001P00173I0068.wav_1567509749_我家朵朵是世界上最漂亮的朵朵。。知道自己是什么样的人。要做什么。无需活在别人非议或期待里。你勤奋.npy')
    data = np.load(inpath)
    data = data.T
    print(data.shape)
    end_idx = find_silences(data, min_silence_sec=0.5, hop_silence_sec=0.2)
    print(end_idx, len(data))
    out_dir = Path(r'data/syns')
    for i, pair in enumerate(zip(end_idx[:-1], end_idx[1:]), 1):
        a, b = pair
        wav = inv_mel_spectrogram(data[a[-1]: b[0]].T, hparams)
        save_wav(wav, out_dir.joinpath(f'sil-{i:02d}.wav'), hparams.sample_rate)
    plt.imshow(data.T)
    plt.colorbar()
    plt.show()



if __name__ == "__main__":
    print(__file__)
    # run_stcmds2librispeechformat()
    # run_create_trans()
    # run_create_json()
    # run_audio_strip()
    run_abspath_train_txt()
    # run_embed_sim()
    # run_get_speakers()
    # run_mel_strip()
