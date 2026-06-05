
    const text = document.querySelector("#text");
    window.addEventListener("resize", (e) => {
        text.innerHTML = "Virtual keyboard detected!!!";
      });

var app = this;
app.hasKeyboard = false;
this.keyboardPress = function() {
    app.hasKeyboard = true;
    $(window).unbind("keyup", app.keyboardPress);
    localStorage.hasKeyboard = true;
    console.log("has keyboard!")
}
$(window).on("keyup", app.keyboardPress)
if(localStorage.hasKeyboard) {
    app.hasKeyboard = true;
    $(window).unbind("keyup", app.keyboardPress);
    console.log("has keyboard from localStorage")
}


if ('ontouchstart' in document.documentElement) {
    // show icon -- detect a touch screen? With the ontouchstart event;
}

const inputField = document.querySelector(".my-input");

const virtualKeyboardDetected = () => alert("Virtual keyboard detected!");

inputField.addEventListener("focusin", () => {
    window.addEventListener("resize", virtualKeyboardDetected )
})
inputField.addEventListener("focusout", () => {
    window.removeEventListener("resize", virtualKeyboardDetected )
})
