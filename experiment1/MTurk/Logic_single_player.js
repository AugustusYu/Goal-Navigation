// Function to create a grid layout
function createGrid() {
    for (let row = 0; row < gridSize.rows; row++) {
        for (let col = 0; col < gridSize.columns; col++) {
            const gridItem = document.createElement("div");
            gridItem.className = "grid-item";

            // Mark the start position
            if (row === startPosition.row && col === startPosition.col) {
                gridItem.classList.add("start");
            } 
            // Mark the end position
            else if (row === endPosition.row && col === endPosition.col) {
                gridItem.classList.add("end");
                gridItem.innerHTML = '&#9733;'; // Add a star symbol at the end position
            }

            // Store the row and column as data attributes
            gridItem.dataset.row = row;
            gridItem.dataset.col = col;

            // Append the grid item to the grid container
            gridContainer.appendChild(gridItem);
        }
    }
}

// Function to add random block obstacles to the grid
function addRandomBlocks() {
    const numBlocks = Math.floor(Math.random() * (gridSize.rows * gridSize.columns));

    for (let i = 0; i < numBlocks; i++) {
        const randomRow = Math.floor(Math.random() * gridSize.rows);
        const randomCol = Math.floor(Math.random() * gridSize.columns);

        // Select a random grid item and add the block class
        const blockItem = document.querySelector(`.grid-item[data-row="${randomRow}"][data-col="${randomCol}"]`);
        blockItem.classList.add("block");
    }
}

// Function to add blocks based on a predefined layout
function addBlocks() {
    for (let i = 0; i < currentLayout.length; i++) {
        for (let j = 0; j < currentLayout[i].length; j++) {
            if (currentLayout[i][j] !== 0) {
                const blockItem = document.querySelector(`.grid-item[data-row="${i}"][data-col="${j}"]`);
                blockItem.classList.add("block");
            }
        }
    }
}

// Function to move the plane based on the given direction
function movePlane(direction, action = -1) {
    const newRow = currentPosition.row + direction.row;
    const newCol = currentPosition.col + direction.col;

    // Check if the new position is within bounds and not blocked
    if (
        newRow >= 0 && newRow < gridSize.rows &&
        newCol >= 0 && newCol < gridSize.columns &&
        !document.querySelector(`.grid-item[data-row="${newRow}"][data-col="${newCol}"]`).classList.contains("block")
    ) {
        // Reset the current position's class
        const currentGridItem = document.querySelector(`.grid-item[data-row="${currentPosition.row}"][data-col="${currentPosition.col}"]`);
        currentGridItem.className = "grid-item";

        // Update the current position
        currentPosition = { row: newRow, col: newCol };

        // Mark the new position as the start (plane position)
        const newGridItem = document.querySelector(`.grid-item[data-row="${newRow}"][data-col="${newCol}"]`);
        newGridItem.className = "grid-item";
        newGridItem.classList.add('start');

        // Track user actions
        user_actions[cur_game].push(action);
        num_actions[cur_game] += 1;

        // Check if the maximum number of actions has been reached
        if (num_actions[cur_game] >= max_action) {
            document.getElementById('button_next_game').style.display = 'block';
        }

        // Check if the plane has reached the end position
        if (currentPosition.row === endPosition.row && currentPosition.col === endPosition.col) {
            document.removeEventListener('keydown', handleKeyPress);
            document.getElementById('button_next_game').style.display = 'block';
        }
    } else {
        // Log invalid action if necessary
        // console.log('Invalid action:', newRow, newCol);
    }
}

// Function to handle keyboard inputs for moving the plane
function handleKeyPress(event) {
    switch (event.key) {
        case "ArrowUp":
        case 'W': case 'w':
            movePlane({ row: -1, col: 0 }, 0);
            break;
        case "ArrowDown":
        case 'S': case 's':
            movePlane({ row: 1, col: 0 }, 1);
            break;
        case "ArrowLeft":
        case 'A': case 'a':
            movePlane({ row: 0, col: -1 }, 2);
            break;
        case "ArrowRight":
        case 'D': case 'd':
            movePlane({ row: 0, col: 1 }, 3);
            break;
        default:
            break;
    }
}
