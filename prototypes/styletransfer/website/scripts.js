var API_ENDPOINT = "https://qgu3stesgg.execute-api.eu-west-1.amazonaws.com/dev"

window.onload = function() {
    document.getElementById('style_choice').onchange = function(e) {
        var img = new Image();
        img.crossOrigin = "anonymous"
        img.onload = draw;
        img.onerror = failed;
        img.src = "https://s3-eu-west-1.amazonaws.com/visualneurons.com/Kand.jpeg";
    };

    function draw() {
        var canvas = document.getElementById('style_img');
        canvas.width = this.width;
        canvas.height = this.height;
        var ctx = canvas.getContext('2d');
        ctx.drawImage(this, 0, 0);
    }

    function failed() {
        alert("The provided file couldn't be loaded as an Image media");
    };
};

document.getElementById('inp').onchange = function(e) {
    var img = new Image();
    img.crossOrigin = "anonymous"
    img.onload = draw;
    img.onerror = failed;
    img.src = URL.createObjectURL(this.files[0]);
};

function draw() {
    var canvas = document.getElementById('content_img');
    canvas.width = this.width;
    canvas.height = this.height;
    var ctx = canvas.getContext('2d');
    ctx.drawImage(this, 0, 0);
}

function failed() {
    alert("The provided file couldn't be loaded as an Image media");
};

document.getElementById("st").onclick = function() {
    //var canvas = document.getElementById("content_img")
    //var image = canvas.toDataURL()
    //var inputData = {"data": "pinging API"};

    $.ajax({
        url: API_ENDPOINT,
        type: 'POST',
        crossDomain: true,
        //data: JSON.stringify(inputData),
        dataType: 'json',
        contentType: "application/json",
        success: openStyleTransferSocket,
    });
}

function openStyleTransferSocket(response) {
    var msg = JSON.parse(response);
    //var webSocketURL = "wss://" + msg.dns + ":8000/styletransfer"
    var webSocketURL = "ws://localhost:8000/styletransfer"

    console.log("openWSConnection::Connecting to: " + webSocketURL);
    try {
        webSocket = new WebSocket(webSocketURL);

        webSocket.onopen = function(openEvent) {
            console.log("WebSocket OPEN: " + JSON.stringify(openEvent, null, 4));
        };

        webSocket.onclose = function(closeEvent) {
            console.log("WebSocket CLOSE: " + JSON.stringify(closeEvent, null, 4));
        };

        webSocket.onerror = function(errorEvent) {
            console.log("WebSocket ERROR: " + JSON.stringify(errorEvent, null, 4));
        };

        webSocket.onmessage = function(messageEvent) {
            if (messageEvent == null) {
                webSocket.close()
            }

            console.log(messageEvent)
                //messageEvent = JSON.stringify(messageEvent)
            var msg = JSON.parse(messageEvent.data);
            console.log(msg.state)

            switch (msg.state) {
                case "model_loaded":
                    console.log("WebSocket STATE: " + msg.state);
                    var to_send = {
                        action: "request_image",
                        data: {
                            content_image: document.getElementById("content_img").toDataURL(),
                            style_image: document.getElementById("style_img").toDataURL(),
                        }
                    };
                    webSocket.send(JSON.stringify(to_send));
                    break;

                case "end_iteration":
                    console.log("WebSocket STATE: " + msg.state);
                    document.getElementById("iteration_img").src = msg.data.image;
                    break;

                default:
                    console.log("WebSocket MESSAGE: " + msg);
            }
        };
    } catch (exception) {
        console.error(exception);
    }
}