import React, { useState } from 'react'
import '../../styles/global.scss'
import { Dna } from  'react-loader-spinner'
import './Enter.scss'
import logo from '../../assets/inccsight.png'

// Icones
import { TbHome2, TbQuestionCircle, TbBrandGithub, TbNews, TbSettings } from 'react-icons/tb'
import View from './View'
import Loading from '../Loading/Loading'

function Enter() {

    localStorage.setItem("folders", JSON.stringify([]));
    
    const [page, setPage] = useState("Input")

    function handleClick(icon, name) {
        const icons = document.querySelectorAll('.enter-icon');

        icons.forEach((i) => {
            if (i === icon) {
                i.classList.add('active');
            } else {
                i.classList.remove('active');
            }
        });

        setPage(name)
    }

    return (
        <div className='enter-container'>
            <div className='loading-screen' id='loading-screen'>
                <Loading/>
            </div>

            <div className='enter-header'>

                <img src={logo} className='icon-logo' alt='Logo do inCCsight' />
                <span className='icon-text'>InCCsight</span>
            </div>

            <div className='enter-body'>

                <div className='enter-left'>

                    <TbHome2 className="enter-icon active" onClick={(e) => handleClick(e.target, "Input")}/>
                    <TbQuestionCircle className="enter-icon" onClick={(e) => handleClick(e.target, "Help")}/>
                    <TbBrandGithub className="enter-icon" onClick={(e) => handleClick(e.target, "Github")}/>
                    <TbNews className="enter-icon" onClick={(e) => handleClick(e.target, "News")}/>
                    <TbSettings className="enter-icon" onClick={(e) => handleClick(e.target, "Settings")}/>
                    
                </div>

                <View type={page}/>

            </div>

        </div>
    )
}

export default Enter