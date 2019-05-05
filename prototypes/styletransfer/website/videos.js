var answers = ["The_Scream.jpg",
    "The_Scream.jpg",
    "Kand1.jpeg",
    "Kand2.png",
    "Monet.jpg",
    "Picasso.png",
    "VanGogh.png"
];

function choose(choices) {
    var index = Math.floor(Math.random() * choices.length);
    return choices[index];
}

window.onload = loadImageInCanvas(choose(answers), document.getElementById('style_img'))

document.getElementById('style_choice').onchange = function(e) {
    loadImageInCanvas(document.getElementById("style_choice").value, document.getElementById('style_img'));
};

AWS.config.region = 'eu-west-1'; // Region

AWS.config.credentials = new AWS.CognitoIdentityCredentials({
    IdentityPoolId: 'eu-west-1:daac3c5a-13e3-4c7d-80d8-869eacaa0f83',
});

var bucketName = 'visualneurons.com-videos'; // Enter your bucket name
var bucket = new AWS.S3({
    params: {
        Bucket: bucketName
    }
});

var fileChooser = document.getElementById('file-chooser');
var button = document.getElementById('upload-button');
var results = document.getElementById('results');

function validateEmail(email) {
    var re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return re.test(email);
}

button.addEventListener('click', function() {
    this.disabled = true;
    button.style.backgroundColor = "#ffc477"
    button.textContent = "Uploaded"
    email = document.getElementById('email').value.toLowerCase();

    if (validateEmail(email) == false) {
        results.textContent = "Please enter a valid email address and try again!"
        setTimeout(function() {
            results.textContent = ''
        }, 2000);
        return
    }

    var file = fileChooser.files[0];

    if (file) {

        results.textContent = '';
        clean_email = email.replace(/[^a-zA-Z0-9]/g, '')
        var objKey = clean_email + '_' + file.name;
        var params = {
            Key: objKey,
            ContentType: file.type,
            Body: file,
            Metadata: {
                'email': email,
                'style': document.getElementById("style_choice").value,
            },
        };

        bucket.putObject(params, function(err, data) {
            if (err) {
                results.textContent = 'ERROR: ' + err;
            } else {
                results.textContent = "Video ingested successfully!";
            }
        });
    } else {
        results.textContent = 'Nothing to upload.';
    }
}, false);

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