import React, { useEffect, useState } from 'react'
import Plot from 'react-plotly.js'
import './VolumetricView.scss'

const fs = window.require('fs')
const zlib = window.require('zlib')
const path = window.require('path')

/**
 * Parses a NIfTI-1 buffer (already decompressed) and returns header + voxel data.
 * Handles datatypes: UINT8(2), INT16(4), INT32(8), FLOAT32(16), FLOAT64(64).
 * NIfTI-1 spec: https://nifti.nimh.nih.gov/pub/dist/src/niftilib/nifti1.h
 */
function parseNifti1(buf) {
    const nx = buf.readInt16LE(40 + 1 * 2)
    const ny = buf.readInt16LE(40 + 2 * 2)
    const nz = buf.readInt16LE(40 + 3 * 2)

    const dx = Math.abs(buf.readFloatLE(76 + 1 * 4))
    const dy = Math.abs(buf.readFloatLE(76 + 2 * 4))
    const dz = Math.abs(buf.readFloatLE(76 + 3 * 4))

    const datatype   = buf.readInt16LE(70)
    const vox_offset = Math.floor(buf.readFloatLE(108))
    const n          = nx * ny * nz
    const off        = buf.byteOffset + vox_offset

    let voxels
    if      (datatype === 2)  voxels = Float32Array.from(new Uint8Array  (buf.buffer, off, n))
    else if (datatype === 4)  voxels = Float32Array.from(new Int16Array  (buf.buffer, off, n))
    else if (datatype === 8)  voxels = Float32Array.from(new Int32Array  (buf.buffer, off, n))
    else if (datatype === 16) voxels = new Float32Array                  (buf.buffer, off, n)
    else if (datatype === 64) voxels = Float32Array.from(new Float64Array(buf.buffer, off, n))
    else                      voxels = new Float32Array                  (buf.buffer, off, n)

    return { nx, ny, nz, dx, dy, dz, voxels }
}

/**
 * Builds x/y/z/value flat arrays for Plotly isosurface.
 * Subsamples the volume by `step` in each axis to keep it responsive.
 * Index formula: idx = ix + iy*nx + iz*nx*ny  (NIfTI column-major)
 */
function buildIsosurfaceArrays(nifti, step = 4) {
    const { nx, ny, nz, dx, dy, dz, voxels } = nifti

    const numX = Math.ceil(nx / step)
    const numY = Math.ceil(ny / step)
    const numZ = Math.ceil(nz / step)
    const n = numX * numY * numZ

    const x = new Float32Array(n)
    const y = new Float32Array(n)
    const z = new Float32Array(n)
    const value = new Float32Array(n)

    let i = 0
    for (let iz = 0; iz < nz; iz += step) {
        for (let iy = 0; iy < ny; iy += step) {
            for (let ix = 0; ix < nx; ix += step) {
                x[i] = ix * dx
                y[i] = iy * dy
                z[i] = iz * dz
                value[i] = voxels[ix + iy * nx + iz * nx * ny]
                i++
            }
        }
    }

    // Plotly expects regular JS arrays
    return {
        x: Array.from(x),
        y: Array.from(y),
        z: Array.from(z),
        value: Array.from(value)
    }
}

function VolumetricView({ filePath }) {
    const [isoData, setIsoData] = useState(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)
    const [opacity, setOpacity] = useState(0.8)

    useEffect(() => {
        setLoading(true)
        setError(null)
        setIsoData(null)

        const load = async () => {
            try {
                const compressed = await fs.promises.readFile(filePath)
                const buf = zlib.gunzipSync(compressed)
                const nifti = parseNifti1(buf)
                const iso = buildIsosurfaceArrays(nifti, 2)
                setIsoData(iso)
            } catch (e) {
                setError(`Não foi possível carregar o arquivo: ${e.message}`)
            } finally {
                setLoading(false)
            }
        }

        load()
    }, [filePath])

    if (loading) {
        return (
            <div className='volumetric-loading'>
                <span>Carregando volume 3D...</span>
            </div>
        )
    }

    if (error) {
        return (
            <div className='volumetric-error'>
                <span>{error}</span>
                <code>{filePath}</code>
            </div>
        )
    }

    const trace = {
        type: 'isosurface',
        x: isoData.x,
        y: isoData.y,
        z: isoData.z,
        value: isoData.value,
        isomin: 0.5,
        isomax: 1.0,
        surface: { count: 2, fill: 0.9, pattern: 'odd' },
        colorscale: [
            [0, '#1a3a6e'],
            [1, '#636EFA']
        ],
        showscale: false,
        opacity: opacity,
        caps: {
            x: { show: false },
            y: { show: false },
            z: { show: false }
        },
        hovertemplate: 'X: %{x:.1f}mm<br>Y: %{y:.1f}mm<br>Z: %{z:.1f}mm<extra>CC (CNN)</extra>'
    }

    const layout = {
        title: {
            text: 'Corpo Caloso — Segmentação Volumétrica (CNN)',
            font: { size: 15 }
        },
        scene: {
            xaxis: { title: 'X (mm)', backgroundcolor: '#e8ecf7', gridcolor: 'white' },
            yaxis: { title: 'Y (mm)', backgroundcolor: '#e8ecf7', gridcolor: 'white' },
            zaxis: { title: 'Z (mm)', backgroundcolor: '#d0d8f0', gridcolor: 'white' },
            bgcolor: '#f0f4ff',
            camera: {
                eye: { x: 1.8, y: 1.8, z: 0.8 }
            },
            aspectmode: 'data'
        },
        height: 560,
        margin: { t: 50, b: 10, l: 10, r: 10 },
        paper_bgcolor: '#f8f9ff'
    }

    return (
        <div className='volumetric-container'>
            <Plot
                data={[trace]}
                layout={layout}
                config={{ displayModeBar: true, displaylogo: false }}
            />

            <div className='volumetric-controls'>
                <label>Opacidade:</label>
                <input
                    type='range'
                    min={0.1}
                    max={1.0}
                    step={0.05}
                    value={opacity}
                    onChange={e => setOpacity(parseFloat(e.target.value))}
                />
                <span>{Math.round(opacity * 100)}%</span>
            </div>
        </div>
    )
}

export default VolumetricView
