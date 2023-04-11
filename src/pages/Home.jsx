import React, { useState } from 'react'

/* Data */
import subjects from '../data/mydata.json'

/* Imagens */
import logo from '../assets/images/inccsight.png'
import unicamp from '../assets/images/unicamp.png'
import miclab from '../assets/images/miclab.png'

/* Componentes */
import SubjectCard from '../components/SubjectCard/SubjectCard'
import ConfigModal from '../components/ConfigModal/ConfigModal'
import View from '../components/View/View'

/* Icones */
import { BsGear } from 'react-icons/bs'
import { createRoot } from 'react-dom/client';

import '../styles/home.scss'

function showConfigs() {
    const container = document.querySelector('#modalArea');
    const root = createRoot(container); // createRoot(container!) if you use TypeScript

    root.render(<ConfigModal root={root} />)
}

function Home() {

    const [filter, setFilter] = useState("")
    const [data, setData] = useState(subjects)
    const [view, setView] = useState("2D")
    console.log(data)

    function filterSubject() {
        let value = document.querySelector("#filter").value
        setFilter(value)
    }

    function changeView(view) {
        let tab2d = document.querySelector("#tab2D")
        let tab3d = document.querySelector("#tab3D")

        if (view === "2D") {
            if (tab2d.classList.contains("active") === false) {
                tab2d.classList.toggle("active")
                tab3d.classList.toggle("active")
                setView("2D")
            }
        } else if (view === "3D") {
            if (tab3d.classList.contains("active") === false) {
                tab3d.classList.toggle("active")
                tab2d.classList.toggle("active")
                setView("3D")
            }
        }
    }

    /* Função que seleciona um sujeito com o Card */

    function selectSubject(name) {
        let subjectPainel = document.querySelector("#subjectPainel");

        if (name === "All") {
            setData(subjects)
        } else {
            subjectPainel.style.display = "flex"
            const selecteds = []
            // Se der erro, tirar o return
            data.filter((subject) => {
                if (subject["Id"] === name) {
                    selecteds.push(subject);
                    return true;  // adiciona o elemento ao novo array
                }
                return false; // descarta o elemento do novo array
            });
            setData(selecteds);
        }
    }

    return (
        <div className='container-home' id="main-area">

            <div id="modalArea">
            </div>

            <div className='header'>

                <div className='banner'>

                    <div className='banner-logos'>
                        <img src={unicamp} className='banner-logo' alt="Logo da Unicamp" />
                        <img src={miclab} className='banner-logo' alt="Logo do Miclab" />
                    </div>

                    <img src={logo} className="img-logo" alt="Logo do inCCsight" />

                    <span className='banner-span'>This is data exploration and visualization tool for diffusion tensor images of the corpus callosum. Upload data folders to begin. Further information can be found here.</span>

                    <div className='banner-selects'>

                        <div className='input-group'>
                            <label>Category: </label>
                            <select>
                                <option>Method</option>
                                <option>Folder</option>
                            </select>
                        </div>

                        <div className='input-group'>
                            <label>Segm. Method: </label>
                            <select>
                                <option>ROQS</option>
                                <option>Watershed</option>
                            </select>
                        </div>

                    </div>

                    <button className='btn-check'>Quality Check<span className='btn-tag'>0</span></button>

                </div>

                <div className='subjects-list'>

                    <label>Subjects</label>

                    <input placeholder='E.g: Subject_00002' id="filter" onChange={filterSubject} />

                    <div className='subjects'>
                        <SubjectCard name="All" onClick={selectSubject} />

                        {data.map((subject, index) => {
                            if (subject["Id"].includes(filter)) {
                                return (
                                    <SubjectCard name={subject["Id"]} id={index} key={index} onClick={selectSubject} />
                                )
                            }
                            return false
                        })}

                    </div>

                </div>

                <div className='square-field'>
                    <div>
                        <span className="qnt">{data.length}</span>
                        <span className="label">Subjects</span>
                    </div>
                </div>

                <BsGear className='gear-icon' onClick={showConfigs} />

            </div>

            <div className='tab-change'>

                <div id="tab2D" className='tab active' onClick={() => changeView("2D")}>Midsagital Segmentation (2D Section)</div>
                <div id="tab3D" className='tab' onClick={() => changeView("3D")}>Volumetric Segmentation (Section 3D)</div>

            </div>

            <View view={view} data={data} />

        </div>
    )
}

export default Home
