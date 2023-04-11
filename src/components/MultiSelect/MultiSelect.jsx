import React, { useState, useEffect } from "react";
import { MultiSelect } from "react-multi-select-component";
import './MultiSelect.scss';

const options = [
    { label: "ROQS Based", value: "ROQS" },
    { label: "CNN Based", value: "CNN"},
    { label: "Watershed (Maintenance)", value: "Watershed", disabled: true },
    { label: "Coming Soon", value: "", disabled: true },
];

const Example = () => {
    const [selected, setSelected] = useState([]);
    
    const handleSelectedChange = (newSelected) => {
        setSelected(newSelected);
        saveToLocalStorage(newSelected);
    }

    const saveToLocalStorage = (selectedOptions) => {
        const labels = selectedOptions.map(option => option.label);
        localStorage.setItem("Methods", JSON.stringify(labels));
    }

    useEffect(() => {
        const storedLabels = localStorage.getItem("selectedOptions");
        if (storedLabels) {
            const parsedLabels = JSON.parse(storedLabels);
            const storedOptions = options.filter(option => parsedLabels.includes(option.label));
            setSelected(storedOptions);
        }
    }, []);

    return (    
        <div className="multiselect-container">
            <h4>Select Methods</h4>
            <MultiSelect
                options={options}
                value={selected}
                onChange={handleSelectedChange}
                labelledBy="Select"
                className="select"
            />
        </div>
    );
};

export default Example;
