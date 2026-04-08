import React, { useState } from 'react'
import Plot from 'react-plotly.js'
import InfoTool from '../../components/InfoTool/InfoTool'
import './TableSegmentation.scss'
import { TbEyeFilled, TbEyeOff } from 'react-icons/tb'

const SCALARS = ["FA", "MD", "RD", "AD"]
const SCALARS_WITH_STD = ["FA", "FA StdDev", "MD", "MD StdDev", "RD", "RD StdDev", "AD", "AD StdDev"]

const METHOD_OPTIONS = [
    { label: "ROQS",          key: "ROQS_scalar"        },
    { label: "Watershed-Based", key: "Watershed_scalar" },
    { label: "CNN-Based",     key: "santarosa_scalars"  },
]

function getMeanValues(subjects, method, scalar) {
    const values = subjects.map(s => s[method][scalar])
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

/** Computes per-column min/max colors for a 2-D array (rows × cols). */
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

function ExpandableSubjectTable({ allSubjects, color, type }) {
    const [open,        setOpen]        = useState(false)
    const [expandMethod, setExpandMethod] = useState(
        type === "3D" ? "santarosa_scalars" : "ROQS_scalar"
    )

    const scalarCols = ["FA", "MD", "RD", "AD"]

    const rows = allSubjects.map(s => {
        const m = s[expandMethod] || {}
        return scalarCols.map(sc =>
            m[sc] != null ? Number(m[sc]).toFixed(6) : "—"
        )
    })

    const cellColors = colorsForRows(rows, scalarCols.length)

    const availableMethods = type === "3D"
        ? METHOD_OPTIONS.filter(m => m.key === "santarosa_scalars")
        : METHOD_OPTIONS

    function exportExpanded() {
        const headers = ["Subject", ...scalarCols]
        const data = allSubjects.map((s, i) => [s["Id"], ...rows[i]])
        const csv  = [headers, ...data].map(r => r.join(',')).join('\n')
        const blob = new Blob([csv], { type: 'text/csv' })
        const url  = URL.createObjectURL(blob)
        const a    = document.createElement('a'); a.href = url
        a.download = `segmentation_subjects_${expandMethod}.csv`; a.click()
        URL.revokeObjectURL(url)
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
                            <label className={color}>Method: </label>
                            <select value={expandMethod} onChange={e => setExpandMethod(e.target.value)}>
                                {availableMethods.map(m => (
                                    <option key={m.key} value={m.key}>{m.label}</option>
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
                                    {scalarCols.map(sc => <th key={sc}>{sc}</th>)}
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

function TableSegmentation(props) {
    const [showStd, setShowStd] = useState(false)
    const [selectedId, setSelectedId] = useState('__all__')

    const allSubjects = props.data
    const subjects = selectedId === '__all__'
        ? allSubjects
        : allSubjects.filter(s => s["Id"] === selectedId)

    const headers = showStd ? ["Method", ...SCALARS_WITH_STD] : ["Method", ...SCALARS]
    const scalarKeys = showStd ? SCALARS_WITH_STD : SCALARS

    const methodNames = ["ROQS", "Watershed-Based", "CNN-Based"]
    const methodKeys  = ["ROQS_scalar", "Watershed_scalar", "santarosa_scalars"]

    let cols = [methodNames]
    for (const key of scalarKeys) {
        const colValues = methodKeys.map(m => getMeanValues(subjects, m, key))
        cols.push(colValues)
    }

    const cellColors = [
        Array(3).fill('#f0f0f0'),
        ...cols.slice(1).map((colValues, i) => {
            const isStdCol = showStd && (i % 2 === 1)
            return isStdCol ? Array(3).fill('white') : getColumnColors(colValues)
        })
    ]

    const layout = {
        height: showStd ? 160 : 130,
        margin: { t: 10, b: 0, l: 10, r: 10 },
        paper_bgcolor: props.bg_color,
        autosize: true,
    }

    if (props.type === "2D") {
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
                        Segmentation Data <InfoTool text="Comparison of the mean values obtained by segmentation in each method." />
                    </span>
                    <button className='btn-export' onClick={() => exportCSV(headers, cols, 'segmentation_data.csv')}>
                        Export
                    </button>
                </div>

                <Plot data={plotData} layout={layout}
                    config={{ responsive: true }}
                    style={{ width: '100%' }}
                    useResizeHandler />

                <div className='options-row'>
                    <div className='select-group'>
                        <label className={props.color}>Std. Dev: </label>
                        <button onClick={() => setShowStd(v => !v)} className="btn-icon">
                            {showStd ? <TbEyeOff /> : <TbEyeFilled />}
                        </button>
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

                <ExpandableSubjectTable allSubjects={allSubjects} color={props.color} type="2D" />
            </div>
        )
    }

    if (props.type === "3D") {
        const cols3d = [["CNN-Based"]]
        for (const key of scalarKeys) {
            cols3d.push([getMeanValues(subjects, "santarosa_scalars", key)])
        }

        const cellColors3d = [
            ['#f0f0f0'],
            ...cols3d.slice(1).map(() => ['white'])
        ]

        const plotData3d = [{
            type: "table",
            header: {
                values: headers,
                align: ["center"],
                line: { width: 1, color: 'black' },
                fill: { color: "grey" },
                font: { family: "Arial", size: 14, color: "white" }
            },
            cells: {
                values: cols3d,
                height: 30,
                align: ["center"],
                line: { width: 1, color: 'black' },
                fill: { color: cellColors3d },
                font: { family: "Arial", size: 12, color: "black" }
            }
        }]

        return (
            <div className='table-field'>
                <div className='table-row'>
                    <span className={`table-title ${props.color}`}>
                        Segmentation Data <InfoTool text="Comparison of the mean values obtained by segmentation in each method." />
                    </span>
                    <button className='btn-export' onClick={() => exportCSV(headers, cols3d, 'segmentation_3d_data.csv')}>
                        Export
                    </button>
                </div>

                <Plot data={plotData3d} layout={layout}
                    config={{ responsive: true }}
                    style={{ width: '100%' }}
                    useResizeHandler />

                <div className='options-row'>
                    <div className='select-group'>
                        <label className={props.color}>Std. Dev: </label>
                        <button onClick={() => setShowStd(v => !v)} className="btn-icon">
                            {showStd ? <TbEyeOff /> : <TbEyeFilled />}
                        </button>
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

                <ExpandableSubjectTable allSubjects={allSubjects} color={props.color} type="3D" />
            </div>
        )
    }
}

export default TableSegmentation
