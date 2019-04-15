var API_ENDPOINT = "https://qgu3stesgg.execute-api.eu-west-1.amazonaws.com/dev"

window.onload = function () {

  document.getElementById('style_choice').onchange = function (e) {
    loadImageInCanvas("Kand.jpeg", document.getElementById('style_img'));
  };

  document.getElementById('inp').onchange = function (e) {
    loadImageInCanvas(URL.createObjectURL(this.files[0]), document.getElementById('content_img'));

  }

  document.getElementById("st").onclick = function () {

    $.ajax({
      url: API_ENDPOINT,
      type: 'POST',
      crossDomain: true,
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

      webSocket.onopen = function (openEvent) {
        console.log("WebSocket OPEN: " + JSON.stringify(openEvent, null, 4));
      };

      webSocket.onclose = function (closeEvent) {
        console.log("WebSocket CLOSE: " + JSON.stringify(closeEvent, null, 4));
      };

      webSocket.onerror = function (errorEvent) {
        console.log("WebSocket ERROR: " + JSON.stringify(errorEvent, null, 4));
      };

      webSocket.onmessage = function (messageEvent) {
        if (messageEvent == null) {
          webSocket.close()
        }

        console.log(messageEvent)
        var msg = JSON.parse(messageEvent.data);
        console.log(msg.state)

        switch (msg.state) {
          case "model_loaded":
            console.log("WebSocket STATE: " + msg.state);
            var to_send = {
              action: "request_image",
              data: {
                content_image: base64FromCanvasId("content_img"),
                style_image: base64FromCanvasId("style_img"),
              }
            };
            webSocket.send(JSON.stringify(to_send));
            break;

          case "end_iteration":
            console.log("WebSocket STATE: " + msg.state);
            document.getElementById("iteration_img").src = "data:image/png;base64," + msg.data.image;
            break;

          default:
            console.log("WebSocket MESSAGE: " + msg);
        }
      };
    } catch (exception) {
      console.error(exception);
    }
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
      canvas.width = this.width;
      canvas.height = this.height;
      var ctx = canvas.getContext('2d');
      ctx.drawImage(this, 0, 0);
    }

    function failed() {
      alert("The provided file couldn't be loaded as an Image media");
    };

  }
}