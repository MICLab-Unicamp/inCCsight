import React, { useState } from 'react'
import Plot from 'react-plotly.js'
import './Scatter.scss'

const COLORS = {
    ROQS: "#636EFA",
    Watershed: "#EF553B",
    CNN: "#3A3A3A"
}

function getAllPoints(data, method, scalar) {
    return data.map(subject => subject[method][scalar])
}

function linearRegression(x, y) {
    const n = x.length
    const sx = x.reduce((a, b) => a + b, 0)
    const sy = y.reduce((a, b) => a + b, 0)
    const sxy = x.reduce((s, xi, i) => s + xi * y[i], 0)
    const sx2 = x.reduce((s, xi) => s + xi * xi, 0)
    const sy2 = y.reduce((s, yi) => s + yi * yi, 0)
    const slope = (n * sxy - sx * sy) / (n * sx2 - sx * sx)
    const intercept = (sy - slope * sx) / n
    const denom = Math.sqrt((n * sx2 - sx * sx) * (n * sy2 - sy * sy))
    const r = denom === 0 ? 0 : (n * sxy - sx * sy) / denom
    return { slope, intercept, r2: r * r }
}

function makeRegressionTrace(x, y, color, name) {
    if (x.length < 2) return null
    const { slope, intercept, r2 } = linearRegression(x, y)
    const xMin = Math.min(...x)
    const xMax = Math.max(...x)
    return {
        x: [xMin, xMax],
        y: [slope * xMin + intercept, slope * xMax + intercept],
        mode: 'lines',
        name: `${name} (R²=${r2.toFixed(3)})`,
        line: { color, width: 2, dash: 'dash' },
        hoverinfo: 'skip'
    }
}

function Scatter(props) {
    const [scalarX, setScalarX] = useState("FA")
    const [scalarY, setScalarY] = useState("MD")

    const sameScalar = scalarX === scalarY
    const ids = props.data.map(s => s["Id"])

    const xWatershed = getAllPoints(props.data, "Watershed_scalar", scalarX)
    const yWatershed = getAllPoints(props.data, "Watershed_scalar", scalarY)
    const xROQS = getAllPoints(props.data, "ROQS_scalar", scalarX)
    const yROQS = getAllPoints(props.data, "ROQS_scalar", scalarY)
    const xCNN = getAllPoints(props.data, "santarosa_scalars", scalarX)
    const yCNN = getAllPoints(props.data, "santarosa_scalars", scalarY)

    const scatterData = [
        {
            x: xROQS, y: yROQS,
            mode: "markers", type: "scatter", name: "ROQS",
            text: ids, hovertemplate: '<b>%{text}</b><br>%{x:.6f} / %{y:.6f}<extra>ROQS</extra>',
            marker: { color: COLORS.ROQS, size: 8 }
        },
        {
            x: xWatershed, y: yWatershed,
            mode: "markers", type: "scatter", name: "Watershed",
            text: ids, hovertemplate: '<b>%{text}</b><br>%{x:.6f} / %{y:.6f}<extra>Watershed</extra>',
            marker: { color: COLORS.Watershed, size: 8 }
        },
        {
            x: xCNN, y: yCNN,
            mode: "markers", type: "scatter", name: "CNN",
            text: ids, hovertemplate: '<b>%{text}</b><br>%{x:.6f} / %{y:.6f}<extra>CNN</extra>',
            marker: { color: COLORS.CNN, size: 8 }
        }
    ]

    if (!sameScalar) {
        const regROQS = makeRegressionTrace(xROQS, yROQS, COLORS.ROQS, "ROQS")
        const regWatershed = makeRegressionTrace(xWatershed, yWatershed, COLORS.Watershed, "Watershed")
        const regCNN = makeRegressionTrace(xCNN, yCNN, COLORS.CNN, "CNN")
        if (regROQS) scatterData.push(regROQS)
        if (regWatershed) scatterData.push(regWatershed)
        if (regCNN) scatterData.push(regCNN)
    }

    const histogramData = [
        { x: xROQS, type: "histogram", name: "ROQS", opacity: 0.5, marker: { color: COLORS.ROQS } },
        { x: xWatershed, type: "histogram", name: "Watershed", opacity: 0.5, marker: { color: COLORS.Watershed } },
        { x: xCNN, type: "histogram", name: "CNN", opacity: 0.5, marker: { color: COLORS.CNN } }
    ]

    const histogramDataY = [
        { x: yROQS, type: "histogram", name: "ROQS", opacity: 0.5, marker: { color: COLORS.ROQS } },
        { x: yWatershed, type: "histogram", name: "Watershed", opacity: 0.5, marker: { color: COLORS.Watershed } },
        { x: yCNN, type: "histogram", name: "CNN", opacity: 0.5, marker: { color: COLORS.CNN } }
    ]

    const scatterLayout = {
        plot_bgcolor: '#E5ECF6',
        height: 700,
        margin: { t: sameScalar ? 40 : 10 },
        yaxis: { gridcolor: 'rgb(255, 255, 255)', title: scalarY },
        xaxis: { gridcolor: 'rgb(255, 255, 255)', title: scalarX },
        legend: { orientation: "h" },
        annotations: sameScalar ? [{
            x: 0.5, y: 0.5, xref: 'paper', yref: 'paper',
            text: 'X e Y são o mesmo escalar',
            showarrow: false,
            font: { size: 16, color: '#EF553B' },
            bgcolor: 'rgba(255,255,255,0.8)',
            bordercolor: '#EF553B',
            borderwidth: 1
        }] : []
    }

    return (
        <div className='scatter-container'>
            <span className='scatter-title'>Segmentation Statistics</span>

            <div className='select-row'>
                <div className='select-scalar'>
                    <span>Scalar X</span>
                    <select className='select' onChange={e => setScalarX(e.target.value)}>
                        {["FA", "MD", "RD", "AD"].map(s => <option key={s} value={s}>{s}</option>)}
                    </select>
                </div>
                <div className='select-scalar'>
                    <span>Scalar Y</span>
                    <select className='select' onChange={e => setScalarY(e.target.value)} defaultValue="MD">
                        {["FA", "MD", "RD", "AD"].map(s => <option key={s} value={s}>{s}</option>)}
                    </select>
                </div>
            </div>

            <div className='scatter-row'>
                <Plot data={scatterData} layout={scatterLayout} />

                <div className='scatter-col'>
                    <Plot
                        data={histogramData}
                        layout={{ barmode: "overlay", width: 800, height: 280, margin: { t: 30, b: 0 }, title: `Scalar: ${scalarX}` }}
                    />
                    <Plot
                        data={histogramDataY}
                        layout={{ barmode: "overlay", width: 800, height: 280, margin: { t: 30, b: 0 }, title: `Scalar: ${scalarY}` }}
                    />
                </div>
            </div>
        </div>
    )
}

export default Scatter
