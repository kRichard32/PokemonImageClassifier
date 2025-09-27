import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import {UploadButton, SubmitButton, ClassifyButton, ClassificiationText} from "./UploadButton.jsx";

function Header() {
  const [count, setCount] = useState(0);
  const [classification, setClassification] = useState(null);
    return (
      <>
          <div className="outline section_height" style={{height: "10vh"}}>
          </div>
          <div className="outline section_height" style={{height: "60vh"}}>
              <Middle setClassification={(event) => setClassification(event)}></Middle>
          </div>
          <div className="outline section_height" style={{height: "20vh"}}>
          <ClassificiationText classification={classification}></ClassificiationText>
          </div>
      </>
  )
}
function Middle({setClassification}) {
    const [selectedFile, setSelectedFile] = useState(null);
    return (
        <>
            <div className="outline half_width" style={{width: `calc(70% - 2px)`}}>
                <ImagePanel selectedFile={selectedFile}></ImagePanel>
            </div>
            <div className="outline half_width" style={{width: `calc(30% - 2px)`}}>
                <RightPanel setClassification={setClassification} selectedFile={selectedFile}
                            setSelectedFile={setSelectedFile}></RightPanel>
            </div>

        </>
    )
}
function ImagePanel({selectedFile}){
    if (selectedFile != null){
        return (
                <img src={URL.createObjectURL(selectedFile)} style={{height:`100%`}}  alt="This is an image" />
        )
    } else {
        return null;
    }
}
function RightPanel({setClassification, selectedFile, setSelectedFile}){

    const [selectedId, setSelectedId] = useState(null);
    return (
        <>
        <div className="outline section_height" style={{height: "calc(30% - 2px)"}}>
            <ClassifyButton selectedId={selectedId} setClassification={setClassification}></ClassifyButton>
        </div>
        <div className="outline section_height" style={{height: "calc(40% - 2px)"}}>
            <UploadButton setSelectedFile={
                (event) => {setSelectedFile(event.target.files[0])}}></UploadButton>
        </div>
        <div className="outline section_height" style={{height: "calc(30% - 2px)"}}>
            <SubmitButton selectedFile={selectedFile} setSelectedId={(event) => {setSelectedId(event)}}> </SubmitButton>
        </div>
        </>
    )
}
export default Header
