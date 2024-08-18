
const url = "http://127.0.0.1:8000/"

document.getElementById("ImageUpload").addEventListener("change", function(e){
    const file = e.target.files[0];
    if (file){
        const reader = new FileReader();
        reader.onload = function(e){
            document.getElementById("UploadedImage").src = e.target.result;
        }
        reader.readAsDataURL(file);
    }
});

async function Recognize(){
    const file = document.getElementById("ImageUpload").files[0];
    const formData = new FormData();

    formData.append("image", file);
    const response = await fetch(
        url+"evaluate/",{
            method: "POST",
            body: formData,
        }
    );
    result = await response.json();
    document.getElementById("CatOrDog").innerText = result.result;
}