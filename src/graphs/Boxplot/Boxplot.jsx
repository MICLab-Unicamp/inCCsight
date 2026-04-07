import React from 'react'
import Plot from 'react-plotly.js'

const COLORS = {
    ROQS: "#636EFA",
    Watershed: "#EF553B",
    CNN: "#3A3A3A"
}

function Boxplot(props) {
    const ids = props.ids || []

    let data = [
        {
            y: props.roqs,
            type: "box",
            name: "ROQS",
            marker: { color: COLORS.ROQS },
            boxpoints: 'all',
            jitter: 0.3,
            pointpos: 0,
            text: ids,
            hovertemplate: '<b>%{text}</b><br>Value: %{y:.6f}<extra>ROQS</extra>'
        },
        {
            y: props.watershed,
            type: "box",
            name: "Watershed",
            marker: { color: COLORS.Watershed },
            boxpoints: 'all',
            jitter: 0.3,
            pointpos: 0,
            text: ids,
            hovertemplate: '<b>%{text}</b><br>Value: %{y:.6f}<extra>Watershed</extra>'
        }
    ]

    if (props.cnn) {
        data.push({
            y: props.cnn,
            type: "box",
            name: "CNN",
            marker: { color: COLORS.CNN },
            boxpoints: 'all',
            jitter: 0.3,
            pointpos: 0,
            text: ids,
            hovertemplate: '<b>%{text}</b><br>Value: %{y:.6f}<extra>CNN</extra>'
        })
    }

    let layout = {
        title: props.title,
        height: 420,
        width: props.width,
        margin: { t: 40, l: 50, r: 10 },
        legend: { orientation: "h" },
        plot_bgcolor: '#E5ECF6',
        yaxis: {
            gridcolor: 'rgb(255, 255, 255)',
            zerolinecolor: 'rgb(255, 255, 255)',
        }
    }

    return <Plot data={data} layout={layout} />
}

export default Boxplot
