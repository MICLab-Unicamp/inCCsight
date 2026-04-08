import React from 'react'
import Plot from 'react-plotly.js'

/* Componentes */
import TableSegmentation from '../../graphs/Table/TableSegmentation'
import TableParcellation from '../../graphs/Table/TableParcellation'
import BoxplotSegmentation from '../../graphs/Boxplot/BoxplotSegmentation'
import BoxplotParcellation from '../../graphs/Boxplot/BoxplotParcellation'
import Scatter from '../../graphs/Scatter/Scatter'
import Midline from '../../graphs/Line/Midline'
import VolumetricView from '../../graphs/Volume/VolumetricView'

/* Icones */
import {AiOutlineClose} from 'react-icons/ai'

import '../../styles/home.scss'
import Radar from '../../graphs/Radar/Radar'

const path = window.require('path')
const fs   = window.require('fs')

function SegmentationPlot({ imgPath }) {
    if (!imgPath) return <span className='msg-image'>Imagem não disponível</span>

    let base64 = null
    try {
        base64 = fs.readFileSync(imgPath).toString('base64')
    } catch (_) {
        return <span className='msg-image'>Imagem não encontrada</span>
    }

    return (
        <Plot
            data={[{
                type: 'image',
                source: `data:image/png;base64,${base64}`,
                hovertemplate: 'x: %{x}  y: %{y}<extra></extra>',
            }]}
            layout={{
                margin: { l: 0, r: 0, t: 0, b: 0 },
                xaxis: { visible: false, showgrid: false },
                yaxis: { visible: false, showgrid: false },
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%', height: '100%' }}
            useResizeHandler
        />
    )
}

/**
 * Resolve o caminho do arquivo NIfTI da segmentação CNN para um sujeito.
 * Prioridade:
 *   1. <cwd>/methods/CNNBased/temp/<id>/inCCsight/cnnBased.nii.gz  (saída real do CNN)
 *   2. <cwd>/methods/CNNBased/temp/<id>/santarosa.nii.gz            (dado de teste)
 *   3. Fallback para dado de demonstração do repositório
 */
function resolveNiftiPath(subjectId) {
    const base = process.cwd()
    const cnnPath  = path.join(base, 'methods', 'CNNBased', 'temp', subjectId, 'inCCsight', 'cnnBased.nii.gz')
    const testPath = path.join(base, 'methods', 'CNNBased', 'temp', subjectId, 'santarosa.nii.gz')

    if (fs.existsSync(cnnPath))  return cnnPath
    if (fs.existsSync(testPath)) return testPath

    return path.join(base, 'methods', 'CNNBased', 'temp', '000215', 'inCCsight', 'cnnBased.nii.gz')
}

function View(props) {

    function closeSelect(){
        let subjectPainel = document.querySelector("#subjectPainel");
        subjectPainel.style.display = "none"
    }

    let data = props.data

    if(props.view === "2D"){
        return (
            <div className='view-container' id="main-area">

                <div className='subject-select' id="subjectPainel">

                    <div className='subject-image'>
                        <span className='subject-name'>{data[0]["Id"]}</span>

                        <div className='image'>
                            <SegmentationPlot imgPath={data[0]["img_path"]} />
                        </div>

                        <div className='image-prompts'>
                            <div className='image-inputs'>

                                <div className='input-group'>
                                    <label>Segm. Method</label>
                                    <select>
                                        <option value="">Watershed</option>
                                        <option value="">ROQS Based</option>
                                        <option value="">CNN Based</option>
                                    </select>
                                </div>

                                <div className='input-group'>
                                    <label>Scalar</label>
                                    <select>
                                        <option value="wFA">wFA</option>
                                        <option value="FA">FA</option>
                                        <option value="MD">MD</option>
                                        <option value="RD">RD</option>
                                        <option value="AD">AD</option>
                                    </select>
                                </div>

                            </div>

                            <div className='image-buttons'>
                                <button className='btn-remove'>Remove</button>
                            </div>

                        </div>
                    </div>

                    <div className='subject-tables'>
                        <TableSegmentation data={data} bg_color="#1F2C56" color="white" type="2D"/>
                        <TableParcellation data={data} bg_color="#1F2C56" color="white" type="2D"/>
                    </div>

                    <AiOutlineClose className='close-icon' onClick={closeSelect}/>

                </div>

                <div className='area-view'>

                    <div className='area-table'>
                        <TableSegmentation data={data} type="2D"/>
                        <TableParcellation data={data} type="2D"/>
                    </div>

                    <div className='area-boxplot'>
                        <BoxplotSegmentation data={data} />
                        <BoxplotParcellation data={data} />
                    </div>

                    <div className='area-scatter'>
                        <Scatter data={data}/>
                    </div>

                    <div className='area-midline'>
                        <Midline data={data}/>
                        <Radar data={data}/>
                    </div>

                </div>

            </div>
        )
    } else if(props.view === "3D"){

        const niftiPath = resolveNiftiPath(data[0]["Id"])

        return(
            <div className='view-container' id="main-area">

                <div className='subject-select' id="subjectPainel">

                    <div className='subject-image'>
                        <span className='subject-name'>3D: {data[0]["Id"]}</span>

                        <div className='image-prompts'>
                            <div className='image-inputs'>

                                <div className='input-group'>
                                    <label>Segm. Method</label>
                                    <select>
                                        <option value="">CNN Based</option>
                                    </select>
                                </div>

                            </div>
                        </div>
                    </div>

                    <div className='subject-tables'>
                        <TableSegmentation data={data} bg_color="#1F2C56" color="white" type="3D"/>
                        <TableParcellation data={data} bg_color="#1F2C56" color="white" type="3D"/>
                    </div>

                    <AiOutlineClose className='close-icon' onClick={closeSelect}/>

                </div>

                <div className='area-view'>

                    <div className='area-table'>
                        <TableSegmentation data={data} type="3D"/>
                        <TableParcellation data={data} type="3D"/>
                    </div>

                    <div className='area-volumetric'>
                        <VolumetricView filePath={niftiPath} />
                    </div>

                </div>

            </div>
        )
    }

}

export default View
