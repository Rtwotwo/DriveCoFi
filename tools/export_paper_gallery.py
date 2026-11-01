from pathlib import Path
import json, hashlib, argparse
import numpy as np
import pypdfium2 as pdfium
from pypdf import PdfReader
from PIL import Image

parser = argparse.ArgumentParser(description='Export standalone web images and saved CSA trajectories')
parser.add_argument('--figures', type=Path, required=True, help='Appendix Figures directory')
parser.add_argument('--csa-cache', type=Path, required=True, help='Directory containing five saved NPZ sets')
args = parser.parse_args()
SOURCE = args.figures.resolve()
SITE = Path(__file__).resolve().parents[1] / 'site'
OUT = SITE / 'assets' / 'gallery'
OUT.mkdir(parents=True, exist_ok=True)
sources = []

def render(src, name):
    doc = pdfium.PdfDocument(src)
    page = doc[0]
    image = page.render(scale=1600 / page.get_width()).to_pil().convert('RGB')
    image.save(OUT / name, 'WEBP', quality=91, method=6)
    sources.append({'image': f'assets/gallery/{name}', 'source': str(src.relative_to(SOURCE)).replace('\\','/'), 'sha256': hashlib.sha256(src.read_bytes()).hexdigest()})
    page.close()
    doc.close()
    return f'./assets/gallery/{name}'

scenes = []
for group, code in [('Straight', 2), ('Left turn', 3), ('Right turn', 1)]:
    slug = group.lower().replace(' ', '-')
    for i in range(1, 4):
        item = {'id': f'{slug}-{i}', 'group': group, 'name': f'{group} · Scene {i}', 'source': f'Appendix C, {group.lower()} comparison'}
        for public, old in [('drivor','drivor'),('cofidrive','more')]:
            item[public] = render(SOURCE / 'cases' / f'c{code}_s{i}_{old}.pdf', f'{slug}-{i}-{public}.webp')
        scenes.append(item)
tokens = sorted(p.name.split('_')[0] for p in (SOURCE / 'attentions' / 'SceneToken').glob('token*.pdf'))
for i, token in enumerate(tokens, 1):
    item = {'id':f'additional-{i}', 'group':'Additional', 'name':f'Additional maneuver · Scene {i}', 'source':'Appendix C, candidate generation and selection'}
    for public, old in [('drivor','drivoR'),('cofidrive','SceneToken')]:
        src = next((SOURCE / 'attentions' / old).glob(f'{token}_*.pdf'))
        item[public] = render(src, f'additional-{i}-{public}.webp')
    scenes.append(item)

attention = {}
for camera, suffix in [('Front','f0'),('Left','l0'),('Right','r0'),('Rear','b0')]:
    attention[camera] = {}
    for public, old in [('drivor','drivoR'),('cofidrive','SceneToken')]:
        src = SOURCE / 'attentions' / old / f'attention_maps_sampled_cam_{suffix}.pdf'
        page = PdfReader(src).pages[0]
        frames = [x for x in page.images if x.image.size == (413,232)]
        assert len(frames) == 6
        attention[camera][public] = []
        for idx, frame in enumerate(frames):
            name = f'attention-{suffix}-{public}-{idx}.webp'
            frame.image.convert('RGB').save(OUT / name, 'WEBP', quality=95, method=6)
            attention[camera][public].append(f'./assets/gallery/{name}')
        sources.append({'images':attention[camera][public], 'source': str(src.relative_to(SOURCE)).replace('\\','/'), 'sha256': hashlib.sha256(src.read_bytes()).hexdigest(), 'operation':'Extract six embedded RGB/attention panels; exclude the colorbar image'})

csa = []
for i, src in enumerate(sorted(args.csa_cache.glob('*.npz')),1):
    a = np.load(src, allow_pickle=False)
    assert a['proposals_64'].shape == (64,8,3)
    assert a['augmented_192'].shape == (192,8,3)
    item = {'id':f'case-{i}', 'name':f'Cached scene {i}', 'initial_speed':round(float(a['v0']),2)}
    for key, name in [('proposals_64','original'),('augmented_192','augmented'),('selected','selected'),('gt','reference')]:
        data = a[key][...,:2]
        assert np.isfinite(data).all()
        item[name] = np.round(data.astype(float),4).tolist()
    csa.append(item)
    sources.append({'data':item['id'], 'source':f'viz/output/exp_arg_traj256/cache/{src.name}', 'sha256':hashlib.sha256(src.read_bytes()).hexdigest(), 'operation':'Retain x/y coordinates; round to four decimal places; no regenerated trajectories'})

data = SITE / 'data'
data.mkdir(exist_ok=True)
for name, obj in [('gallery',{'scenes':scenes,'attention':attention}),('csa',csa),('sources',sources)]:
    (data / f'{name}.json').write_text(json.dumps(obj,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
print(f'Prepared {len(scenes)} paired scenes, 48 attention panels and {len(csa)} saved CSA sets.')
