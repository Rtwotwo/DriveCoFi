from pathlib import Path
import json, html

site=Path(__file__).resolve().parents[1]/'site'
out=site/'assets'/'images'
scene=json.loads((site/'data/csa.json').read_text())[0]
allpts=[p for traj in scene['original']+scene['augmented']+[scene['selected'],scene['reference']] for p in traj]+[[0,0]]
xs=[-p[1] for p in allpts]; ys=[p[0] for p in allpts]
xmin,xmax=min(xs)-3,max(xs)+3;ymin,ymax=min(ys)-3,max(ys)+3
scale=min(490/(xmax-xmin),360/(ymax-ymin));cx=(xmin+xmax)/2;cy=(ymin+ymax)/2
parts=['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="500" viewBox="0 0 1200 500" role="img" aria-label="Saved original proposals and CSA training candidates"><rect width="1200" height="500" rx="16" fill="#f2f7fa"/><g font-family="Arial,sans-serif" fill="#17364b">']
for panel, title in enumerate(['Original proposals · 64','Auxiliary training list · 64 + 192']):
    offset=panel*600
    parts.append(f'<text x="{offset+300}" y="36" text-anchor="middle" font-size="23" font-weight="700">{title}</text>')
    def xy(p):return (offset+300+(-p[1]-cx)*scale,250-(p[0]-cy)*scale)
    def path(points,color,width,opacity,dots=False):
        pts=[xy(p) for p in [[0,0]]+points]
        parts.append(f'<polyline points="{" ".join(f"{x:.2f},{y:.2f}" for x,y in pts)}" fill="none" stroke="{color}" stroke-width="{width}" opacity="{opacity}"/>')
        if dots:
            for x,y in pts[1:]:parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}" stroke="white"/>')
    if panel:
        for t in scene['augmented']:path(t,'#9273b5',1,.3)
    for t in scene['original']:path(t,'#197aaf',1.3,.45)
    path(scene['reference'],'#268577',2,1)
    path(scene['selected'],'#bd7e00',3,1,True)
    x,y=xy([0,0]);parts.append(f'<rect x="{x-6:.2f}" y="{y-9:.2f}" width="12" height="18" rx="3" fill="#17364b"/>')
    # Scale bar in data units.
    parts.append(f'<path d="M{offset+50} 410v-5h{scale*5:.2f}v5" fill="none" stroke="#536e80"/><text x="{offset+50}" y="432" font-size="13">5 m · equal axis scale</text>')
parts.append('<text x="600" y="478" text-anchor="middle" font-size="17">Saved CSA cache · 8 future poses · Click to explore layers, scenes and time steps</text></g></svg>')
(out/'csa-preview.svg').write_text(''.join(parts),encoding='utf-8')

rows=json.loads((site/'data/results.json').read_text())['ablations'][0]['rows']
parts=['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="325" viewBox="0 0 1000 325" role="img" aria-label="Sequential ablation: base 91.70, TC 91.80, CCS 91.90, CSA 92.40 PDMS"><rect width="1000" height="325" rx="16" fill="#f2f7fa"/><g font-family="Arial,sans-serif" fill="#17364b"><text x="35" y="38" font-size="24" font-weight="700">Sequential component ablation</text><text x="35" y="64" font-size="14">Train85k / navtest · 10 epochs · 64 inference candidates</text>']
for score in [91,91.5,92,92.5,93]:
    x=280+(score-91)/2*580
    parts.append(f'<path d="M{x} 93V265" stroke="#d3e0e8"/><text x="{x}" y="289" text-anchor="middle" font-size="14">{score:.1f}</text>')
for i,(name,value) in enumerate(rows):
    y=115+i*44;x=280+(value-91)/2*580
    parts.append(f'<text x="35" y="{y+5}" font-size="17">{html.escape(name)}</text><circle cx="{x}" cy="{y}" r="7" fill="#176b99"/><text x="930" y="{y+5}" text-anchor="end" font-size="20" font-weight="700">{value:.2f}</text>')
parts.append('<text x="570" y="314" text-anchor="middle" font-size="13">PDMS axis 91–93 · reported point estimates</text></g></svg>')
(out/'ablation-summary.svg').write_text(''.join(parts),encoding='utf-8')
print('Created two data-driven SVG previews.')
