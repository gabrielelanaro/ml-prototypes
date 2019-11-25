var API_ENDPOINT = "https:50vllx1cci.execute-api.eu-west-1.amazonaws.com/prod"
var prompt = document.getElementById('hugging-prompt');
var samples = document.getElementById('how-many-samples');
var words = document.getElementById('how-many-words');
var temperature = document.getElementById('temperature');
var nucleus = document.getElementById('nucleus');
var topn = document.getElementById('top-n');
var button = document.getElementById('hugging');

function truncatePrompt(prompt){
    prompt = prompt.value.substring(0,100)
    index = prompt.lastIndexOf(" ")
    return prompt.substring(0, index)
}

function validate(str, min, max) {
    n = parseFloat(str);
    return (!isNaN(n) && n >= min && n <= max);
  }

function processResponse(response){
    console.log("Ok!!!!")
    console.log(response)
    button.disabled = false;
    button.style.backgroundColor = "#FF9900"
    button.textContent = "Let the machine take over!"
}

button.addEventListener('click', function() {

    if(!validate(samples.value, 1, 4)) {
        alert("The number of text samples must be between 1 and 4. You have selected ".concat(samples.value, "!"));
        return false;
    }
    if(!validate(words.value, 1, 100)) {
        alert("The number of words must be between 1 and 100. You have selected ".concat(words.value, "!"));
        return false;
    }
    if(!validate(temperature.value, 0, 1)) {
        alert("Temperature must be between 0 and 1. You have selected ".concat(temperature.value, "!"));
        return false;
    }
    if(!validate(nucleus.value, 0, 1)) {
        alert("A probability must be between 0 and 1. In nucleus filtering you have typed ".concat(nucleus.value, "!"));
        return false;
    }
    if(!validate(topn.value, 0, 1000)) {
        alert("You can select between 1 and 1000 top N words. You have typed ".concat(topn.value, "!"));
        return false;
    }
    if(prompt.value.length==0){
        alert("Your text prompt is empty! Please trigger the model with at least one word.");
        return false
    }

    prompt = truncatePrompt(prompt)
    var inputData = {"prompt": prompt,
                    "samples": samples.value,
                    "words": words.value,
                    "temperature": temperature.value,
                    "nucleus": nucleus.value,
                    "topn": topn.value
                }

    console.log(inputData)
    this.disabled = true;
    button.style.backgroundColor = "#ffc477"
    button.textContent = "Running"
    $.ajax({
        url: API_ENDPOINT,
        type: 'POST',
        crossDomain: true,
        tryCount : 0,
        retryLimit : 3,
        dataType: 'json',
        contentType: "application/json",
        data: JSON.stringify(inputData),
        success: processResponse,
        error: function(xhr, status, error) {
            console.log("AJAX status:" + status)
            console.log("retry " + this.tryCount + " of " + this.retryLimit)
            if (status == 'error') {
                this.tryCount++;
                if (this.tryCount <= this.retryLimit) {
                    //try again
                    $.ajax(this);
                    return;
                }
                document.getElementById("results").textContent = "Ouch... Sorry, it seems we ran out of artistic GPUs! Can you try again in a couple of minutes?";            
                return;
            }
            
        }
    });
}, false);