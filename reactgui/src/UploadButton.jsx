import { useState } from 'react'
import reactLogo from './assets/react.svg'
import './App.css'

export function ClassificiationText({classification}){
    if (classification != null){
        return (
            <>
            <h1 className="fg vert_center">
                        Predicted: {classification}
                    </h1>
            </>
        )
    } else {
        return null;
    }
}

export function ClassifyButton({selectedId, setClassification}){
    if (selectedId != null){
        return (
            <>
                <button style={{display: "flex", flex_direction: "row"}}
                        className="logo" onClick={() => classifyFile(selectedId, setClassification)}>
                    <h1 className="fg vert_center">
                        Classify
                    </h1>
                </button>
            </>
        )
    } else {
        return null;
    }
}
const classifyFile = async (selectedId, setClassification) => {
    const response = await fetch(`http://localhost:8080/classify?filePath=${selectedId}`, {
        method: 'GET',
        headers: {
            'Accept': 'application/json',
        }
    });
    const data = await response.json();
    setClassification(data['id'])
    console.log(data['id'])
    return data['id']
}
export function UploadButton({setSelectedFile}) {
    // const onFileChange = (event) => {
    //     setSelectedFile(event.target.files[0]);
    // https://codesandbox.io/p/sandbox/z67yyo0wj3?file=%2Findex.js%3A29%2C9
    // };
    return (
        <>
            <label htmlFor="file-upload" style={{display: "flex", flex_direction: "row"}}
                   className="custom_label label_logo" >
                <div className="fg vert_center">
                    <img src={reactLogo} className={"fg fit_widths"} alt={"my_logo"}/>
                </div>
                <h1 className="fg vert_center">
                    Upload
                </h1>
                <div className="fg vert_center">
                    <img src={reactLogo} className={"fg fit_widths"} alt={"my_logo"}/>
                </div>
            </label>
            <input id="file-upload" type="file" onChange={setSelectedFile} style={{display: "none"}}/>
        </>
    )
}
export function SubmitButton({selectedFile, setSelectedId}) {
    const [count, setCount] = useState(0)

    return (
        <>
            <button style={{display: "flex", flex_direction: "row"}}
                    className="logo" onClick={() => postFile(selectedFile, setSelectedId)}>
                <h1 className="fg vert_center">
                    Submit! 
                </h1>
            </button>
        </>
    )
}
const postFile = async (selectedFile, setSelectedId) => {
    const formData = new FormData();
    formData.append(
        "files",
        selectedFile,
        selectedFile.name
    );
    const response = await fetch('http://localhost:8080',
        {method: "POST",
            body: formData});
    const data = await response.json();
    setSelectedId(data['Success'])
    console.log(data['Success'])
    return data['Success']
}

