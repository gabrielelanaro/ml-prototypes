var API_ENDPOINT = "https://znwm2qyvdi.execute-api.eu-west-1.amazonaws.com/dev"

window.onload = loadImageInCanvas("Kand2.png", document.getElementById('style_img'))

document.getElementById('style_choice').onchange = function(e) {
    loadImageInCanvas(document.getElementById("style_choice").value, document.getElementById('style_img'));
};

document.getElementById('inp').onchange = function(e) {
    loadImageInCanvas(URL.createObjectURL(this.files[0]), document.getElementById('content_img'));
    this.disabled = true;
}

document.getElementById("st").onclick = function() {
    document.getElementById("limit").textContent = "Our GPU artist is on it";
    this.disabled = true;
    document.getElementById("st").style.backgroundColor = "#ffc477"
    document.getElementById("st").value = "Running"

    var inputData = {"data": base64FromCanvasId("content_img"), "style": document.getElementById("style_choice").value}

    $.ajax({
        url: API_ENDPOINT,
        type: 'POST',
        crossDomain: true,
        data: JSON.stringify(inputData),
        dataType: 'json',
        contentType: "application/json",
        success: function (response) {
            if (response.includes("Sorry")){
                document.getElementById("limit").textContent = response;
                return
             }
            else {
                document.getElementById("iteration_img").src = "data:image/png;base64," + response;
                document.getElementById("limit").textContent = "Look at this brand new piece of art!";
            }
          },
    });
}

function base64FromCanvasId(canvas_id) {
    return document.getElementById(canvas_id).toDataURL().split(',')[1];
}

function loadImageInCanvas(url, canvas) {
    var img = $("<img />", {
        src: url,
        crossOrigin: "Anonymous",
    }).load(draw).error(failed);

    function draw() {
        canvas.width = this.width * (300 / this.height);
        canvas.height = 300;
        var ctx = canvas.getContext('2d');
        ctx.drawImage(this, 0, 0, this.width, this.height, 0, 0, this.width * (300 / this.height), 300);
    }

    function failed() {
        alert("The provided file couldn't be loaded as an Image media");
    };

}