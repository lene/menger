#!/usr/bin/env bash
# Caustic-DELTA comparison (Sprint 33 course-correction, 2026-07-03).
#
# Whole-image menger-vs-pbrt comparison is not achievable: pbrt `sppm` carries global
# illumination that menger (direct + caustics only) structurally lacks, and menger's diffuse
# model adds a constant ambient term — neither of which caustic physics affects. So we compare
# the *caustic contribution in isolation*:
#
#     caustic = render(caustics ON) - render(caustics OFF)
#
# The shared direct + ambient (menger) / direct + GI (pbrt) lighting cancels, leaving only the
# caustic. On the pbrt side the OFF render uses the `path` integrator, which cannot connect a
# diffuse floor to the light through specular glass and so is caustic-free.
#
# Usage:
#   compare-caustic-delta.sh <menger_on.pfm> <menger_off.pfm> <pbrt_on.pfm> <pbrt_off.pfm>
#
# Prints caustic energy (Σ luminance delta), peak, energy ratio (menger/pbrt) and the Pearson
# spatial correlation of the two caustic maps. Targets to drive toward as P2–P9 land:
#   energy ratio -> 1.0,  correlation -> >0.8,  peak ratio -> ~1.0.
#
# All four inputs must be linear PFMs at the same resolution (tone map None; pbrt EXR->PFM via
# `imgtool convert --clamp 1`).
set -euo pipefail

on_m="${1:?menger caustics-ON PFM required}"
off_m="${2:?menger caustics-OFF PFM required}"
on_p="${3:?pbrt caustics-ON (sppm) PFM required}"
off_p="${4:?pbrt caustics-OFF (path) PFM required}"

for f in "$on_m" "$off_m" "$on_p" "$off_p"; do
  [ -f "$f" ] || { echo "MISSING: $f" >&2; exit 2; }
done

node - "$on_m" "$off_m" "$on_p" "$off_p" <<'NODE'
const fs=require('fs');
function load(p){const b=fs.readFileSync(p);let i=0,L=[],s='';
  while(L.length<3){const c=String.fromCharCode(b[i++]);if(c==='\n'){L.push(s);s='';}else s+=c;}
  const [w,h]=L[1].split(/\s+/).map(Number);const le=parseFloat(L[2])<0;const off=i;
  const a=new Float32Array(w*h*3);
  for(let k=0;k<a.length;k++)a[k]=le?b.readFloatLE(off+k*4):b.readFloatBE(off+k*4);
  return {w,h,a};}
const [onM,offM,onP,offP]=process.argv.slice(2).map(load);
const N=onM.w*onM.h;
for(const im of [offM,onP,offP]) if(im.w!==onM.w||im.h!==onM.h){console.error('resolution mismatch');process.exit(2);}
function delta(on,off){const d=new Float32Array(N);
  for(let p=0;p<N;p++){const r=on.a[p*3]-off.a[p*3],g=on.a[p*3+1]-off.a[p*3+1],b=on.a[p*3+2]-off.a[p*3+2];
    d[p]=Math.max(0,(r+g+b)/3);} return d;}
const dm=delta(onM,offM), dp=delta(onP,offP);
function stat(d){let s=0,mx=0;for(const v of d){s+=v;if(v>mx)mx=v;}return{sum:s,max:mx};}
const sm=stat(dm), sp=stat(dp);
let mm=0,mp=0;for(let i=0;i<N;i++){mm+=dm[i];mp+=dp[i];}mm/=N;mp/=N;
let smp=0,saa=0,sbb=0;for(let i=0;i<N;i++){const a=dm[i]-mm,b=dp[i]-mp;smp+=a*b;saa+=a*a;sbb+=b*b;}
const corr=smp/Math.sqrt(saa*sbb||1);
const fmt=(x,n=3)=>x.toFixed(n);
console.log(`menger caustic: energy ${fmt(sm.sum,1)}  peak ${fmt(sm.max,4)}`);
console.log(`pbrt   caustic: energy ${fmt(sp.sum,1)}  peak ${fmt(sp.max,4)}`);
console.log(`energy ratio (menger/pbrt): ${fmt(sm.sum/sp.sum)}   [target 1.0]`);
console.log(`peak ratio   (menger/pbrt): ${fmt(sm.max/sp.max)}   [target ~1.0]`);
console.log(`spatial correlation (Pearson): ${fmt(corr)}   [target > 0.8]`);
NODE
