#!/usr/bin/env python3
"""Embed pinned original XeSL; override only the destination declaration/store.

No address, endian or depth conversion function is rewritten here. The pinned
upstream SPIR-V arrays were verified identical to the SDK's legacy binaries.
"""
import argparse
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent / 'texture_load_original/gpu/shaders'

def expand(path):
    text = path.read_text()
    return re.sub(r'^#include "([^"]+)"',
                  lambda m: expand((path.parent / m[1]).resolve()), text, flags=re.M)

def shader(depth=False):
    prefix = '''#version 450
#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_samplerless_texture_functions : require
#define SHADING_LANGUAGE_GLSL_XE 1
#define XE_TEXTURE_LOAD_RESOLUTION_SCALED
'''
    shared = expand(ROOT / 'texture_load.xesli')
    if depth:
        shared += expand(ROOT / 'pixel_formats.xesli')
    store = '''
// Destination-only adaptation. All source addresses and conversion stay XeSL.
#undef array_buffer_wo_declare_xe
#define array_buffer_wo_declare_xe(value_type,name,glsl_set,glsl_binding,hlsl_u,hlsl_u_space)
#undef array_buffer_store_xe
#define array_buffer_store_xe(name,position,value) XeDirectStore(uint(position),value)
layout(set=0,binding=0,FORMAT) writeonly uniform image2D xe_direct_dest;
void XeDirectStore(uint word_index, uvec4 value) {
  uint byte_offset = (word_index << 4u) - push_const_xe(xe_texture_load_host_offset);
  uint pitch = push_const_xe(xe_texture_load_host_pitch);
  uint x = (byte_offset % pitch) >> 2u;
  uint y = byte_offset / pitch;
  ivec2 extent = imageSize(xe_direct_dest);
  for (uint i=0u; i<4u; ++i) {
    if (x+i < uint(extent.x) && y < uint(extent.y)) {
      imageStore(xe_direct_dest, ivec2(x+i,y), DECODE);
    }
  }
}
'''.replace('FORMAT', 'r32f' if depth else 'rgba8')
    store = store.replace('DECODE', 'vec4(uintBitsToFloat(value[i]),0.0,0.0,1.0)' if depth else 'unpackUnorm4x8(value[i])')
    # Include the original entry point, unmodified. Include guards prevent
    # redeclaration of the original shared library already embedded above.
    entry = ROOT / ('texture_load_depth_unorm_scaled.cs.xesl' if depth else 'texture_load_32bpb_scaled.cs.xesl')
    return prefix + shared + store + expand(entry)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('output');args=parser.parse_args()
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    text='#pragma once\n// Generated from pinned, unmodified Xenia XeSL. See texture_load_original/provenance.json.\n'
    for name,depth in [('kDirectStorageImageLoadShaderSource',False),('kDirectStorageImageDepthShaderSource',True)]:
        text += 'static constexpr char '+name+'[] = R"XESL('+shader(depth)+')XESL";\n'
    out.write_text(text)
if __name__=='__main__':main()
