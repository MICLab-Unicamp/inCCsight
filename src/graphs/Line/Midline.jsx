import React, { useState } from 'react'
import Plot from 'react-plotly.js'
import './Midline.scss'

const COLORS = {
    ROQS: "#636EFA",
    Watershed: "#EF553B"
}

// Witelson region boundaries (approximate, on 200-point midline)
const WITELSON_BOUNDARIES = [40, 80, 120, 160]
const WITELSON_LABELS = ["P1", "P2", "P3", "P4", "P5"]
const WITELSON_MIDPOINTS = [20, 60, 100, 140, 180]

function getMeanPoints(data, method, scalar) {
    const size = data[0][method][scalar].length
    return Array.from({ length: size }, (_, p) => {
        const sum = data.reduce((acc, s) => acc + s[method][scalar][p], 0)
        return sum / data.length  // fixed: divide by subject count, not point count
    })
}

function getStdDevPoints(data, method, scalar) {
    const size = data[0][method][scalar].length
    return Array.from({ length: size }, (_, p) => {
        const vals = data.map(s => s[method][scalar][p])
        const mean = vals.reduce((a, b) => a + b, 0) / vals.length
        return Math.sqrt(vals.reduce((sum, v) => sum + (v - mean) ** 2, 0) / vals.length)
    })
}

function getMeanThickness(data, method) {
    const size = data[0][method].length
    return Array.from({ length: size }, (_, p) => {
        const sum = data.reduce((acc, s) => acc + s[method][p], 0)
        return sum / data.length
    })
}

function getStdDevThickness(data, method) {
    const size = data[0][method].length
    return Array.from({ length: size }, (_, p) => {
        const vals = data.map(s => s[method][p])
        const mean = vals.reduce((a, b) => a + b, 0) / vals.length
        return Math.sqrt(vals.reduce((sum, v) => sum + (v - mean) ** 2, 0) / vals.length)
    })
}

function buildBandTraces(yMean, yStd, fillColor, name) {
    const x = Array.from({ length: yMean.length }, (_, i) => i)
    const yUpper = yMean.map((v, i) => v + yStd[i])
    const yLower = yMean.map((v, i) => v - yStd[i])
    return [
        {
            x, y: yUpper,
            mode: 'lines', line: { width: 0 },
            showlegend: false, hoverinfo: 'skip',
            name: `${name} upper`, legendgroup: name
        },
        {
            x, y: yLower,
            fill: 'tonexty', mode: 'lines', line: { width: 0 },
            fillcolor: fillColor,
            showlegend: false, hoverinfo: 'skip',
            name: `${name} lower`, legendgroup: name
        }
    ]
}

function Midline(props) {
    const [scalar, setScalar] = useState("FA")

    const x = Array.from({ length: 200 }, (_, i) => i)

    let traces = []
    let yAxisTitle = scalar

    if (scalar !== "Thickness") {
        const roqsMean = getMeanPoints(props.data, "ROQS_midlines", scalar)
        const roqsStd = getStdDevPoints(props.data, "ROQS_midlines", scalar)
        const wsMean = getMeanPoints(props.data, "Watershed_midlines", scalar)
        const wsStd = getStdDevPoints(props.data, "Watershed_midlines", scalar)

        traces = [
            ...buildBandTraces(roqsMean, roqsStd, 'rgba(99,110,250,0.2)', "ROQS"),
            ...buildBandTraces(wsMean, wsStd, 'rgba(239,85,59,0.2)', "Watershed"),
            {
                x, y: roqsMean,
                mode: 'lines', name: 'ROQS', legendgroup: 'ROQS',
                line: { color: COLORS.ROQS, width: 2 },
                hovertemplate: 'Point %{x}<br>Value: %{y:.6f}<extra>ROQS</extra>'
            },
            {
                x, y: wsMean,
                mode: 'lines', name: 'Watershed', legendgroup: 'Watershed',
                line: { color: COLORS.Watershed, width: 2 },
                hovertemplate: 'Point %{x}<br>Value: %{y:.6f}<extra>Watershed</extra>'
            }
        ]
    } else {
        // Fixed: ROQS uses ROQS_thickness, Watershed uses Watershed_thickness
        const roqsMean = getMeanThickness(props.data, "ROQS_thickness")
        const roqsStd = getStdDevThickness(props.data, "ROQS_thickness")
        const wsMean = getMeanThickness(props.data, "Watershed_thickness")
        const wsStd = getStdDevThickness(props.data, "Watershed_thickness")

        yAxisTitle = "Thickness (mm)"

        traces = [
            ...buildBandTraces(roqsMean, roqsStd, 'rgba(99,110,250,0.2)', "ROQS"),
            ...buildBandTraces(wsMean, wsStd, 'rgba(239,85,59,0.2)', "Watershed"),
            {
                x, y: roqsMean,
                mode: 'lines', name: 'ROQS', legendgroup: 'ROQS',
                line: { color: COLORS.ROQS, width: 2 },
                hovertemplate: 'Point %{x}<br>Thickness: %{y:.4f}<extra>ROQS</extra>'
            },
            {
                x, y: wsMean,
                mode: 'lines', name: 'Watershed', legendgroup: 'Watershed',
                line: { color: COLORS.Watershed, width: 2 },
                hovertemplate: 'Point %{x}<br>Thickness: %{y:.4f}<extra>Watershed</extra>'
            }
        ]
    }

    // Witelson region boundary shapes
    const shapes = WITELSON_BOUNDARIES.map(xPos => ({
        type: 'line',
        x0: xPos, x1: xPos,
        y0: 0, y1: 1,
        yref: 'paper',
        line: { color: 'rgba(0,0,0,0.25)', width: 1, dash: 'dot' }
    }))

    // Region label annotations
    const annotations = WITELSON_MIDPOINTS.map((xPos, i) => ({
        x: xPos, y: 1.02,
        xref: 'x', yref: 'paper',
        text: WITELSON_LABELS[i],
        showarrow: false,
        font: { size: 11, color: '#555' }
    }))

    const layout = {
        title: "Midline Plots",
        height: 420, width: 660,
        margin: { t: 50, l: 50, r: 10 },
        legend: { orientation: "h", x: 1, y: 1.1, xanchor: 'right' },
        plot_bgcolor: '#E5ECF6',
        yaxis: { gridcolor: 'rgb(255,255,255)', title: yAxisTitle },
        xaxis: { gridcolor: 'rgb(255,255,255)', title: "Points Along CC Body" },
        shapes,
        annotations
    }

    return (
        <div className='midline-container'>
            <Plot data={traces} layout={layout} />

            <div className='select-scalar'>
                <span>Scalar</span>
                <select className='select' onChange={e => setScalar(e.target.value)}>
                    {["FA", "MD", "RD", "AD", "Thickness"].map(s => (
                        <option key={s} value={s}>{s}</option>
                    ))}
                </select>
            </div>
        </div>
    )
}

export default Midline
