// Function to create the grid layout
function createGrid() {
    for (let row = 0; row < gridSize.rows; row++) {
        for (let col = 0; col < gridSize.columns; col++) {
            const gridItem = document.createElement("div");
            gridItem.className = "grid-item";

            // Add 'start' class if the current position is the start position
            if (row === startPosition.row && col === startPosition.col) {
                gridItem.classList.add("start");
                gridItem.textContent = "Start"; // Label the start position
            }

            // Store row and column as data attributes
            gridItem.dataset.row = row;
            gridItem.dataset.col = col;

            // Append the grid item to the grid container
            gridContainer.appendChild(gridItem);
        }
    }

    // Add goals to the grid based on goal positions
    for (let i = 0; i < numGoal; i++) {
        const goalItem = document.querySelector(`.grid-item[data-row="${goalPosition[i][0]}"][data-col="${goalPosition[i][1]}"]`);
        
        if (i === 0) {
            goalItem.classList.add("end1"); // Apply class for the first goal
            goalItem.innerHTML = '&#9733;'; // Star symbol for the first goal
        } else {
            goalItem.classList.add("end2"); // Apply class for the second goal
            goalItem.innerHTML = "&#9650;"; // Triangle symbol for the second goal
        }
    }
}

// Function to add blocks (obstacles) to the grid
function addBlocks() {
    // Iterate over the grid layout and add blocks where needed
    for (let i = 0; i < gridSize.rows; i++) {
        for (let j = 0; j < gridSize.columns; j++) {
            if (currentLayout[i][j] == 1) { // Check if the current position is a block
                const blockItem = document.querySelector(`.grid-item[data-row="${i}"][data-col="${j}"]`);
                blockItem.classList.add("block"); // Apply block class
            }
        }
    }
}

// Function to animate the path traversal
function addPath() {
    pathCount = 0;
    
    // Use setInterval to animate the path one step at a time
    intervalID = setInterval(function addOnePath() {
        // Reset the current grid item to a path
        let curobj = document.querySelector(`.grid-item[data-row="${currentPosition.row}"][data-col="${currentPosition.col}"]`);
        curobj.className = "grid-item";
        curobj.classList.add('path');

        // Move to the next position in the path
        currentPosition = { row: pathPosition[pathCount][0], col: pathPosition[pathCount][1] };
        let nextobj = document.querySelector(`.grid-item[data-row="${currentPosition.row}"][data-col="${currentPosition.col}"]`);
        nextobj.className = "grid-item";
        nextobj.classList.add('start');

        pathCount++;

        // Stop the interval when the path is fully traversed
        if (pathCount >= pathPosition.length) {
            document.getElementById('belief-decision').style.display = 'block';
            clearInterval(intervalID);
        }
    }, 1000); // Move every 1 second
}

// Function to replay the path from the start position
function replayPath() {
    currentPosition = startPosition;

    // Reset the start position
    let curobj = document.querySelector(`.grid-item[data-row="${currentPosition.row}"][data-col="${currentPosition.col}"]`);
    curobj.className = "grid-item";
    curobj.classList.add('start');

    // Reset all grid items that were part of the path
    for (let i = 0; i < pathPosition.length; i++) {
        let curobj = document.querySelector(`.grid-item[data-row="${pathPosition[i][0]}"][data-col="${pathPosition[i][1]}"]`);
        curobj.className = "grid-item";
    }

    // Replay the path animation
    addPath();
}

// Placeholder function for moving the plane (to be implemented)
function movePlane(direction, action = -1) {
    // Implement plane movement logic here
}

// Placeholder function for handling key presses (to be implemented)
function handleKeyPress(event) {
    // Implement key press handling logic here
}
