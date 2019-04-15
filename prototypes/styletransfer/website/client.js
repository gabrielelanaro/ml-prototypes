const url = "ws://localhost:8000/styletransfer"
const connection = new WebSocket(url)

connection.onopen = function (event) {
    console.log("client is ready"); 
  };


connection.onmessage = function(event) {
  if (event == null) {
        connection.close()
    }

  var msg = JSON.parse(event);
  
  switch(msg.state) {
    case "model_loaded":
      var to_send = {
          action: "request_image",
          data: {
              content_image: document.getElementById("content_img").toDataURL(),
              style_image: document.getElementById("style_img").toDataURL()
          }
        };
      connection.send(JSON.stringify(to_send));
      break;

    case "end_iteration":
      document.getElementById("iteration_img").src = msg.data.image;
      break;

    default:
      console.log(msg)
  }
  
};
