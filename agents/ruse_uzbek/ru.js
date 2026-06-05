<script>
    const text = document.querySelector("#text");
    window.addEventListener("resize", (e) => {
        text.innerHTML = "Virtual keyboard detected!!!";
      });
</script>


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
