import React, { useState } from 'react'
import Plot from 'react-plotly.js'
import './TableParcellation.scss'
import InfoTool from '../../components/InfoTool/InfoTool'

const SEG_METHOD_OPTIONS = [
    { label: "ROQS",            key: "ROQS_parcellation"       },
    { label: "Watershed-Based", key: "Watershed_parcellation"  },
]

const PARC_METHODS   = ["Witelson", "Hofer", "Chao", "Cover", "Freesurfer"]
const SCALARS        = ["FA", "RD", "AD", "MD"]
const PARTS          = ["P1", "P2", "P3", "P4", "P5"]

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

function colorsForRows(rows, colCount) {
    return Array.from({ length: colCount }, (_, ci) => {
        const col = rows.map(r => Number(r[ci]))
        const max = Math.max(...col)
        const min = Math.min(...col)
        return col.map(v =>
            rows.length < 2  ? 'transparent'
            : v === max      ? 'rgba(144,238,144,0.45)'
            : v === min      ? 'rgba(255,182,193,0.45)'
            :                  'transparent'
        )
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

function ExpandableParcTable({ allSubjects, color }) {
    const [open,        setOpen]        = useState(false)
    const [segMethod,   setSegMethod]   = useState("ROQS_parcellation")
    const [parcMethod,  setParcMethod]  = useState("Witelson")
    const [scalar,      setScalar]      = useState("FA")

    const rows = allSubjects.map(s => {
        const m = s[segMethod] || {}
        return PARTS.map(part => {
            const key = `${parcMethod}_${scalar}_${part}`
            return m[key] != null ? Number(m[key]).toFixed(6) : "—"
        })
    })

    const cellColors = colorsForRows(rows, PARTS.length)

    function exportExpanded() {
        const headers = ["Subject", ...PARTS]
        const data = allSubjects.map((s, i) => [s["Id"], ...rows[i]])
        const csv  = [headers, ...data].map(r => r.join(',')).join('\n')
        const blob = new Blob([csv], { type: 'text/csv' })
        const url  = URL.createObjectURL(blob)
        const a    = document.createElement('a'); a.href = url
        a.download = `parcellation_subjects_${segMethod}_${parcMethod}_${scalar}.csv`
        a.click(); URL.revokeObjectURL(url)
    }

    return (
        <div className='expandable-section'>
            <div className='expandable-header' onClick={() => setOpen(v => !v)}>
                <span>Per-Subject Data</span>
                <span className='expand-icon'>{open ? '▲' : '▼'}</span>
            </div>

            {open && (
                <div className='expandable-content'>
                    <div className='expand-controls'>
                        <div className='select-group'>
                            <label className={color}>Seg. Method: </label>
                            <select value={segMethod} onChange={e => setSegMethod(e.target.value)}>
                                {SEG_METHOD_OPTIONS.map(m => (
                                    <option key={m.key} value={m.key}>{m.label}</option>
                                ))}
                            </select>
                        </div>
                        <div className='select-group'>
                            <label className={color}>Parc. Method: </label>
                            <select value={parcMethod} onChange={e => setParcMethod(e.target.value)}>
                                {PARC_METHODS.map(m => (
                                    <option key={m} value={m}>{m}</option>
                                ))}
                            </select>
                        </div>
                        <div className='select-group'>
                            <label className={color}>Scalar: </label>
                            <select value={scalar} onChange={e => setScalar(e.target.value)}>
                                {SCALARS.map(s => (
                                    <option key={s} value={s}>{s}</option>
                                ))}
                            </select>
                        </div>
                        <button className='btn-export' onClick={exportExpanded}>Export</button>
                    </div>

                    <div className='subject-table-wrap'>
                        <table className='subject-table'>
                            <thead>
                                <tr>
                                    <th>Subject</th>
                                    {PARTS.map(p => <th key={p}>{p}</th>)}
                                </tr>
                            </thead>
                            <tbody>
                                {allSubjects.map((s, ri) => (
                                    <tr key={s["Id"]}>
                                        <td className='subject-id'>{s["Id"]}</td>
                                        {rows[ri].map((val, ci) => (
                                            <td key={ci} style={{ backgroundColor: cellColors[ci][ri] }}>
                                                {val}
                                            </td>
                                        ))}
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}
        </div>
    )
}

function TableParcellation(props) {
    const [methodParcellation, setMethodParcellation] = useState("Witelson")
    const [scalar, setScalar] = useState("FA")
    const [selectedId, setSelectedId] = useState('__all__')

    const headers = ["Method", "P1", "P2", "P3", "P4", "P5"]
    const allSubjects = props.data
    const subjects = selectedId === '__all__'
        ? allSubjects
        : allSubjects.filter(s => s["Id"] === selectedId)
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

                {allSubjects.length > 1 && (
                    <div className='select-group'>
                        <label className={props.color}>Subject: </label>
                        <select value={selectedId} onChange={e => setSelectedId(e.target.value)}>
                            <option value='__all__'>All (mean)</option>
                            {allSubjects.map(s => (
                                <option key={s["Id"]} value={s["Id"]}>{s["Id"]}</option>
                            ))}
                        </select>
                    </div>
                )}
            </div>

            <ExpandableParcTable allSubjects={allSubjects} color={props.color} />
        </div>
    )
}

export default TableParcellation
