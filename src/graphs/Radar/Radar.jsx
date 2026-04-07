import React, { useState } from 'react'
import Plot from 'react-plotly.js'
import './Radar.scss'

const COLORS = {
    ROQS: "#636EFA",
    Watershed: "#EF553B"
}

const PARTS = ['P1', 'P2', 'P3', 'P4', 'P5']

function getMeanValues(subjects, method, parc_method, scalar, part) {
    const name = `${parc_method}_${scalar}_${part}`
    const values = subjects.map(s => s[method][name])
    return parseFloat((values.reduce((a, b) => a + b, 0) / values.length).toFixed(6))
}

function getAllValues(subjects, method, parc_method, scalar) {
    return PARTS.map(part => getMeanValues(subjects, method, parc_method, scalar, part))
}

function normalize(values) {
    const max = Math.max(...values)
    if (max === 0) return values
    return values.map(v => parseFloat((v / max).toFixed(6)))
}

function Radar(props) {
    const [methodRadar, setMethodRadar] = useState("Witelson")
    const [scalarRadar, setScalarRadar] = useState("FA")
    const [normalized, setNormalized] = useState(false)

    let wsValues = getAllValues(props.data, "Watershed_parcellation", methodRadar, scalarRadar)
    let roqsValues = getAllValues(props.data, "ROQS_parcellation", methodRadar, scalarRadar)

    if (normalized) {
        const allVals = [...wsValues, ...roqsValues]
        const globalMax = Math.max(...allVals)
        wsValues = wsValues.map(v => parseFloat((v / globalMax).toFixed(6)))
        roqsValues = roqsValues.map(v => parseFloat((v / globalMax).toFixed(6)))
    }

    // Close the polygon by repeating first value
    const theta = [...PARTS, PARTS[0]]

    const plotData = [
        {
            type: 'scatterpolar',
            r: [...wsValues, wsValues[0]],
            theta,
            fill: 'toself',
            name: "Watershed",
            line: { color: COLORS.Watershed }
        },
        {
            type: 'scatterpolar',
            r: [...roqsValues, roqsValues[0]],
            theta,
            fill: 'toself',
            name: "ROQS",
            line: { color: COLORS.ROQS }
        }
    ]

    const layout = {
        title: "Radar Parcellation",
        legend: { orientation: "h" },
        polar: {
            radialaxis: {
                visible: true,
                title: normalized ? "Normalized" : scalarRadar
            }
        }
    }

    return (
        <div className='radar-container'>
            <Plot data={plotData} layout={layout} />

            <div className='options-col'>
                <div className='select-group'>
                    <label>Parc. Method: </label>
                    <select onChange={e => setMethodRadar(e.target.value)}>
                        {["Witelson", "Hofer", "Chao", "Cover", "Freesurfer"].map(m => (
                            <option key={m} value={m}>{m}</option>
                        ))}
                    </select>
                </div>

                <div className='select-group'>
                    <label>Scalar: </label>
                    <select onChange={e => setScalarRadar(e.target.value)}>
                        {["FA", "RD", "AD", "MD"].map(s => (
                            <option key={s} value={s}>{s}</option>
                        ))}
                    </select>
                </div>

                <div className='select-group'>
                    <label>Normalize (0–1): </label>
                    <input
                        type="checkbox"
                        checked={normalized}
                        onChange={e => setNormalized(e.target.checked)}
                    />
                </div>
            </div>
        </div>
    )
}

export default Radar
