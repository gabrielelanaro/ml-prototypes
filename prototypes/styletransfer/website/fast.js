var API_ENDPOINT = "https://znwm2qyvdi.execute-api.eu-west-1.amazonaws.com/dev"
var button = document.getElementById('st');

window.onload = loadImageInCanvas("Several_Circles.jpg", document.getElementById('style_img'))

var quotes = ["Patience is bitter, but its fruit is sweet. (J.J. Rousseau)",
    "Wise to resolve, and patient to perform. (Homer)",
    "Genius is eternal patience. (Michelangelo)",
    "He that can have patience can have what he will. (B. Franklin)",
    "Humility is attentive patience. (S. Weil)",
    "Patience is passion tamed. (L. Abbott)",
    "Writing is good, thinking is better. Cleverness is good, patience is better. (H. Hesse)",
    "Patience is the art of hoping. (L. de Clapiers)"
];

function choose(choices) {
    var index = Math.floor(Math.random() * choices.length);
    return choices[index];
}

document.getElementById('style_choice').onchange = function(e) {
    loadImageInCanvas(document.getElementById("style_choice").value, document.getElementById('style_img'));
};

document.getElementById('inp').onchange = function(e) {
    loadImageInCanvas(URL.createObjectURL(this.files[0]), document.getElementById('content_img'));
    this.disabled = true;
}

function processResponse(response){
    window.clearInterval(window.intervalID);
    document.getElementById("iteration_img").src = "data:image/png;base64," + response;
    document.getElementById("limit").textContent = "Look at this brand new piece of art!";
    button.disabled = !button.disabled;
    button.style.backgroundColor = "#FF9900"
    button.textContent = "Fast Style Transfer!"
}

function doStuff(){
    document.getElementById("limit").textContent = choose(quotes); 
}

document.getElementById("st").onclick = function() {
    document.getElementById("limit").textContent = "Hold on, checking if we got some deep artists around...";
    button.disabled = true;
    button.style.backgroundColor = "#ffc477"
    button.textContent = "Running"
    window.intervalID = window.setInterval(doStuff, 10000);

    var inputData = {"data": base64FromCanvasId("content_img"), "style": document.getElementById("style_choice").value}

    $.ajax({
        url: API_ENDPOINT,
        type: 'POST',
        crossDomain: true,
        tryCount : 0,
        retryLimit : 3,
        data: JSON.stringify(inputData),
        dataType: 'json',
        contentType: "application/json",
        success: processResponse,
        error: function(xhr, status, error) {
            console.log("AJAX status:" + status)
            console.log("retry " + this.tryCount + " of " + this.retryLimit)
            if (status == 'error') {
                this.tryCount++;
                if (this.tryCount <= this.retryLimit) {
                    //try again
                    $.ajax(this);
                    //if (this.tryCount==1){document.getElementById("limit").textContent = "Found them! </br> Stretching their fingers..."}
                    //if (this.tryCount==2){document.getElementById("limit").textContent = "Finding inspiration..."}
                    //if (this.tryCount==3){document.getElementById("limit").textContent = "It is hard to come up with something cool..."}
                    //if (this.tryCount==4){document.getElementById("limit").textContent = "Don't give up! We are almost there..."}
                    //if (this.tryCount==5){document.getElementById("limit").textContent = "Ok, I admit they are being slow..."}
                    //if (this.tryCount==6){document.getElementById("limit").textContent = "I hear some typing!"}
                    //if (this.tryCount>=7){document.getElementById("limit").textContent = "Still nothing?"}
                    return;
                }
                document.getElementById("limit").textContent = "Ouch... Sorry, it seems we ran out of deep neural artists! Can you try again in a couple of minutes?";            
                return;
            }
            
        }
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