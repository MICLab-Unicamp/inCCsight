import React, { useState, useEffect, useRef } from 'react'
import FolderSelector from '../FolderSelector/FolderSelector'
import { TbPlus } from 'react-icons/tb'
import Question from '../Question/Question'

const { ipcRenderer } = window.require('electron');

const questions = [
    { "question": "How to enter data into the software?", "response": "Click on 'Selecionar pasta' for each group, choose the folder containing the subjects (each subject is a sub-folder with DTI files), then click 'Executar análise'." },
    { "question": "How to add more groups to be analyzed?", "response": "Click the '+ Adicionar grupo' button to add a new row. Give each group a name and select its folder. All groups will be processed together." },
    { "question": "How to compare groups?", "response": "After analysis, the main dashboard shows a group filter at the top of the subjects list. Select a group to filter, or 'Todos' to see all subjects at once." },
    { "question": "How to suggest modifications to the tool?", "response": "Open an issue on the project's GitHub repository." },
    { "question": "What are the current limitations of the tool?", "response": "The tool currently supports DTI data in NIfTI format (.nii / .nii.gz) with eigenvector/eigenvalue files (dti_L1–3, dti_V1–3)." }
]

function openWindow() {
    ipcRenderer.send('open-window');
}

let _nextId = 2; // starts at 2 because the first group is id=1

function View(props) {

    const [folderGroups, setFolderGroups] = useState([
        { id: 1, path: '', groupName: 'Group 1' }
    ]);
    const [filter, setFilter] = useState('');
    const logRef = useRef(null);

    // ── IPC listeners ──────────────────────────────────────────────────────────

    useEffect(() => {
        function onOutput(_, text) {
            const log = document.querySelector('#pipeline-log');
            if (log) {
                log.textContent += text;
                log.scrollTop = log.scrollHeight;
            }
        }

        function onDone(_, { code }) {
            const loadingScreen = document.querySelector('#loading-screen');
            if (code === 0) {
                openWindow();
            } else {
                if (loadingScreen) loadingScreen.style.display = 'none';
                alert('Pipeline encerrou com erros. Verifique o log.');
            }
        }

        ipcRenderer.on('pipeline-output', onOutput);
        ipcRenderer.on('pipeline-done', onDone);

        return () => {
            ipcRenderer.removeListener('pipeline-output', onOutput);
            ipcRenderer.removeListener('pipeline-done', onDone);
        };
    }, []);

    // ── Folder group management ────────────────────────────────────────────────

    function updateGroup(id, updates) {
        setFolderGroups(prev => prev.map(g => g.id === id ? { ...g, ...updates } : g));
    }

    function addGroup() {
        const id = _nextId++;
        setFolderGroups(prev => [
            ...prev,
            { id, path: '', groupName: `Group ${prev.length + 1}` }
        ]);
    }

    function removeGroup(id) {
        setFolderGroups(prev => prev.filter(g => g.id !== id));
    }

    // ── Pipeline actions ───────────────────────────────────────────────────────

    function showLoading() {
        const log = document.querySelector('#pipeline-log');
        if (log) log.textContent = '';
        const screen = document.querySelector('#loading-screen');
        if (screen) screen.style.display = 'flex';
    }

    function startAnalyzes() {
        const valid = folderGroups.filter(g => g.path);
        if (valid.length === 0) {
            alert('Selecione pelo menos uma pasta antes de executar a análise.');
            return;
        }

        const paths = valid.map(g => g.path);
        const groupsMap = {};
        valid.forEach(g => { groupsMap[g.path] = g.groupName || `Group ${g.id}`; });

        showLoading();
        ipcRenderer.send('run-pipeline', { paths, groupsMap, skipCnn: false });
    }

    function loadLast() {
        showLoading();
        ipcRenderer.send('load-last');
    }

    function loadTestData() {
        openWindow();
    }

    // ── Render ─────────────────────────────────────────────────────────────────

    const filteredQuestions = questions.filter(q =>
        q.question.toLowerCase().includes(filter.toLowerCase()) ||
        q.response.toLowerCase().includes(filter.toLowerCase())
    );

    if (props.type === 'Input') {
        return (
            <div className='enter-right'>
                <span className='enter-name'>
                    Selecione as pastas a analisar — uma por grupo.
                </span>

                <div className='folders-inputs'>
                    {folderGroups.map((g, idx) => (
                        <FolderSelector
                            key={g.id}
                            id={g.id}
                            path={g.path}
                            groupName={g.groupName}
                            colorIndex={idx}
                            onUpdate={updates => updateGroup(g.id, updates)}
                            onRemove={folderGroups.length > 1 ? () => removeGroup(g.id) : null}
                        />
                    ))}

                    <button className='add-btn' onClick={addGroup}>
                        <TbPlus className='add-icon' />
                        <span className='add-label'>Adicionar grupo</span>
                    </button>
                </div>

                <div className='row-btns'>
                    <div className='secondary-btns'>
                        <div className='btn-history' onClick={loadLast}>
                            <span>Última análise</span>
                        </div>
                        <div className='btn-demo' onClick={loadTestData}>
                            <span>Dados de teste</span>
                        </div>
                    </div>
                    <button className='btn-start' onClick={startAnalyzes}>
                        Executar análise
                    </button>
                </div>
            </div>
        );
    }

    if (props.type === 'Help') {
        return (
            <div className='enter-question'>
                <div className='search-field'>
                    <span className='enter-name'>Perguntas frequentes sobre a ferramenta.</span>
                    <input
                        className='search-input'
                        placeholder='Ex: Como inserir dados'
                        value={filter}
                        onChange={e => setFilter(e.target.value)}
                    />
                </div>
                <div className='questions-container'>
                    {filteredQuestions.map((q, i) => (
                        <Question key={i} question={q.question} response={q.response} />
                    ))}
                </div>
            </div>
        );
    }

    return (
        <div className='news-container'>
            <span>Coming Soon</span>
        </div>
    );
}

export default View;
