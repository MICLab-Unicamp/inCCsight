import React, { useState } from 'react'
import Plot from 'react-plotly.js'
import './TableParcellation.scss'
import InfoTool from '../../components/InfoTool/InfoTool'

function getMeanValues(subjects, method, parc_method, scalar, part) {
    const name = `${parc_method}_${scalar}_${part}`
    const values = subjects.map(s => s[method][name])
    return (values.reduce((a, b) => a + b, 0) / values.length).toFixed(6)
}

function getColumnColors(colValues) {
    const nums = colValues.map(Number)
    const max = Math.max(...nums)
    const min = Math.min(...nums)
    return nums.map(v => {
        if (v === max) return 'rgba(144, 238, 144, 0.6)'
        if (v === min) return 'rgba(255, 182, 193, 0.6)'
        return 'white'
    })
}

function exportCSV(headers, cols, filename) {
    const rows = [headers.join(',')]
    for (let r = 0; r < cols[0].length; r++) {
        rows.push(cols.map(col => col[r]).join(','))
    }
    const blob = new Blob([rows.join('\n')], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    a.click()
    URL.revokeObjectURL(url)
}

function TableParcellation(props) {
    const [methodParcellation, setMethodParcellation] = useState("Witelson")
    const [scalar, setScalar] = useState("FA")

    const headers = ["Method", "P1", "P2", "P3", "P4", "P5"]
    const subjects = props.data
    const parts = ["P1", "P2", "P3", "P4", "P5"]

    let cols = [["ROQS", "Watershed-Based"]]
    for (const part of parts) {
        const colValues = [
            getMeanValues(subjects, "ROQS_parcellation", methodParcellation, scalar, part),
            getMeanValues(subjects, "Watershed_parcellation", methodParcellation, scalar, part)
        ]
        cols.push(colValues)
    }

    const cellColors = [
        ['#f0f0f0', '#f0f0f0'],
        ...cols.slice(1).map(colValues => getColumnColors(colValues))
    ]

    const layout = {
        height: 130,
        margin: { t: 10, b: 0, l: 10, r: 10 },
        paper_bgcolor: props.bg_color,
        autosize: true,
    }

    const plotData = [{
        type: "table",
        header: {
            values: headers,
            align: ["center"],
            line: { width: 1, color: 'black' },
            fill: { color: "grey" },
            font: { family: "Arial", size: 14, color: "white" }
        },
        cells: {
            values: cols,
            height: 30,
            align: ["center"],
            line: { width: 1, color: 'black' },
            fill: { color: cellColors },
            font: { family: "Arial", size: 12, color: "black" }
        }
    }]

    return (
        <div className='table-field'>
            <div className='table-row'>
                <span className={`table-title ${props.color}`}>
                    Parcellation Data <InfoTool text="Comparison of the average of the values obtained from each part by the parcellation in each method." />
                </span>
                <button className='btn-export' onClick={() => exportCSV(headers, cols, `parcellation_${methodParcellation}_${scalar}.csv`)}>
                    Export
                </button>
            </div>

            <Plot data={plotData} layout={layout}
                config={{ responsive: true }}
                style={{ width: '100%' }}
                useResizeHandler />

            <div className='options-row'>
                <div className='select-group'>
                    <label className={props.color}>Parc. Method: </label>
                    <select onChange={e => setMethodParcellation(e.target.value)}>
                        {["Witelson", "Hofer", "Chao", "Cover", "Freesurfer"].map(m => (
                            <option key={m} value={m}>{m}</option>
                        ))}
                    </select>
                </div>

                <div className='select-group'>
                    <label className={props.color}>Scalar: </label>
                    <select onChange={e => setScalar(e.target.value)}>
                        {["FA", "RD", "AD", "MD"].map(s => (
                            <option key={s} value={s}>{s}</option>
                        ))}
                    </select>
                </div>
            </div>
        </div>
    )
}

export default TableParcellation
