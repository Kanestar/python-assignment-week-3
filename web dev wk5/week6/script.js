// Get the elements
const h1 = document.getElementById("challenges");  // Must match your h1's id
const button = document.getElementById("actionButton");  // Must match your button's id

// Add click handler
button.addEventListener('click', function() {
    h1.textContent = "xenotransplantation";  // This WILL change the heading
    console.log("Button was clicked!");
});
const paragraph = document.getElementById("xeno"); 
paragraph.style.color = "red"; // This will change the color of the paragraph text
paragraph.style.fontSize = "20px"; // This will change the font size of the paragraph text
paragraph.style.fontFamily = "Arial"; // This will change the font family of the paragraph text

const toggleBtn = document.getElementById("toggleParagraph");
let isParagraphVisible = true;
//Add/remove functionality
toggleBtn.addEventListener('click', function() {
    if (isParagraphVisible) {
        paragraph.style.display = "none";
        toggleBtn.textContent = "Show Paragraph";
    } else {
        paragraph.style.display = "block";
        toggleBtn.textContent = "Hide Paragraph";
    }
    isParagraphVisible = !isParagraphVisible;
    console.log("Paragraph visibility toggled");
});