// Neural Network Class for AI Learning
class NeuralNetwork {
    constructor(inputSize, hiddenSize, outputSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        
        // Initialize weights with small random values
        this.weightsIH = this.createMatrix(this.hiddenSize, this.inputSize);
        this.weightsHO = this.createMatrix(this.outputSize, this.hiddenSize);
        this.biasH = this.createMatrix(this.hiddenSize, 1);
        this.biasO = this.createMatrix(this.outputSize, 1);
        
        this.learningRate = 0.1;
        this.momentum = 0.9;
        
        // Previous weight changes for momentum
        this.prevWeightsIH = this.createMatrix(this.hiddenSize, this.inputSize, 0);
        this.prevWeightsHO = this.createMatrix(this.outputSize, this.hiddenSize, 0);
    }
    
    createMatrix(rows, cols, fillValue = null) {
        return Array(rows).fill().map(() => 
            Array(cols).fill().map(() => fillValue !== null ? fillValue : (Math.random() * 2 - 1) * 0.5)
        );
    }
    
    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }
    
    sigmoidDerivative(x) {
        return x * (1 - x);
    }
    
    predict(inputArray) {
        // Convert input to matrix
        const inputs = this.arrayToMatrix(inputArray);
        
        // Hidden layer
        const hidden = this.matrixMultiply(this.weightsIH, inputs);
        this.matrixAdd(hidden, this.biasH);
        this.matrixMap(hidden, this.sigmoid);
        
        // Output layer
        const output = this.matrixMultiply(this.weightsHO, hidden);
        this.matrixAdd(output, this.biasO);
        this.matrixMap(output, this.sigmoid);
        
        return this.matrixToArray(output);
    }
    
    train(inputArray, targetArray) {
        // Forward propagation
        const inputs = this.arrayToMatrix(inputArray);
        
        const hidden = this.matrixMultiply(this.weightsIH, inputs);
        this.matrixAdd(hidden, this.biasH);
        this.matrixMap(hidden, this.sigmoid);
        
        const outputs = this.matrixMultiply(this.weightsHO, hidden);
        this.matrixAdd(outputs, this.biasO);
        this.matrixMap(outputs, this.sigmoid);
        
        // Convert targets to matrix
        const targets = this.arrayToMatrix(targetArray);
        
        // Calculate output errors
        const outputErrors = this.matrixSubtract(targets, outputs);
        
        // Calculate gradients
        const gradients = this.matrixMap(outputs, this.sigmoidDerivative, true);
        this.matrixMultiplyElement(gradients, outputErrors);
        this.matrixMultiplyScalar(gradients, this.learningRate);
        
        // Calculate deltas
        const hiddenT = this.matrixTranspose(hidden);
        const weightsHODeltas = this.matrixMultiply(gradients, hiddenT);
        
        // Update weights and biases with momentum
        for (let i = 0; i < this.weightsHO.length; i++) {
            for (let j = 0; j < this.weightsHO[i].length; j++) {
                const delta = weightsHODeltas[i][j] + this.momentum * this.prevWeightsHO[i][j];
                this.weightsHO[i][j] += delta;
                this.prevWeightsHO[i][j] = delta;
            }
        }
        this.matrixAdd(this.biasO, gradients);
        
        // Calculate hidden layer errors
        const weightsHOT = this.matrixTranspose(this.weightsHO);
        const hiddenErrors = this.matrixMultiply(weightsHOT, outputErrors);
        
        // Calculate hidden gradients
        const hiddenGradient = this.matrixMap(hidden, this.sigmoidDerivative, true);
        this.matrixMultiplyElement(hiddenGradient, hiddenErrors);
        this.matrixMultiplyScalar(hiddenGradient, this.learningRate);
        
        // Calculate input->hidden deltas
        const inputsT = this.matrixTranspose(inputs);
        const weightsIHDeltas = this.matrixMultiply(hiddenGradient, inputsT);
        
        // Update input->hidden weights with momentum
        for (let i = 0; i < this.weightsIH.length; i++) {
            for (let j = 0; j < this.weightsIH[i].length; j++) {
                const delta = weightsIHDeltas[i][j] + this.momentum * this.prevWeightsIH[i][j];
                this.weightsIH[i][j] += delta;
                this.prevWeightsIH[i][j] = delta;
            }
        }
        this.matrixAdd(this.biasH, hiddenGradient);
        
        // Return error for accuracy tracking
        return outputErrors.reduce((sum, row) => sum + Math.abs(row[0]), 0) / outputErrors.length;
    }
    
    // Matrix operations
    arrayToMatrix(arr) {
        return arr.map(val => [val]);
    }
    
    matrixToArray(matrix) {
        return matrix.map(row => row[0]);
    }
    
    matrixMultiply(a, b) {
        const result = this.createMatrix(a.length, b[0].length, 0);
        for (let i = 0; i < a.length; i++) {
            for (let j = 0; j < b[0].length; j++) {
                let sum = 0;
                for (let k = 0; k < a[0].length; k++) {
                    sum += a[i][k] * b[k][j];
                }
                result[i][j] = sum;
            }
        }
        return result;
    }
    
    matrixAdd(a, b) {
        for (let i = 0; i < a.length; i++) {
            for (let j = 0; j < a[0].length; j++) {
                a[i][j] += b[i][j];
            }
        }
    }
    
    matrixSubtract(a, b) {
        const result = this.createMatrix(a.length, a[0].length);
        for (let i = 0; i < a.length; i++) {
            for (let j = 0; j < a[0].length; j++) {
                result[i][j] = a[i][j] - b[i][j];
            }
        }
        return result;
    }
    
    matrixTranspose(matrix) {
        const result = this.createMatrix(matrix[0].length, matrix.length);
        for (let i = 0; i < matrix.length; i++) {
            for (let j = 0; j < matrix[0].length; j++) {
                result[j][i] = matrix[i][j];
            }
        }
        return result;
    }
    
    matrixMultiplyElement(a, b) {
        for (let i = 0; i < a.length; i++) {
            for (let j = 0; j < a[0].length; j++) {
                a[i][j] *= b[i][j];
            }
        }
    }
    
    matrixMultiplyScalar(matrix, scalar) {
        for (let i = 0; i < matrix.length; i++) {
            for (let j = 0; j < matrix[0].length; j++) {
                matrix[i][j] *= scalar;
            }
        }
    }
    
    matrixMap(matrix, fn, returnNew = false) {
        const result = returnNew ? this.createMatrix(matrix.length, matrix[0].length) : matrix;
        matrix.forEach((row, i) => {
            row.forEach((val, j) => {
                result[i][j] = fn(val);
            });
        });
        return result;
    }
}

// Particle class for visual effects
class Particle {
    constructor(x, y, color, velocity) {
        this.x = x;
        this.y = y;
        this.color = color;
        this.velocity = velocity;
        this.life = 1.0;
        this.decay = 0.02;
        this.size = Math.random() * 3 + 1;
    }
    
    update() {
        this.x += this.velocity.x;
        this.y += this.velocity.y;
        this.life -= this.decay;
        this.velocity.x *= 0.98;
        this.velocity.y *= 0.98;
    }
    
    draw(ctx) {
        ctx.save();
        ctx.globalAlpha = this.life;
        ctx.fillStyle = this.color;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
    }
}

// Player Beam class for full charge attack
class PlayerBeam {
    constructor(x, y, directionX, directionY, canvasWidth, canvasHeight, color = '#4a9eff', width = 12) {
        this.startX = x;
        this.startY = y;
        this.directionX = directionX;
        this.directionY = directionY;
        this.color = color;
        this.width = width; // Configurable beam width
        this.life = 1.0;
        this.maxLife = 0.15; // Very short duration (0.15 seconds)
        this.canvasWidth = canvasWidth;
        this.canvasHeight = canvasHeight;
        
        // Convert hex color to RGB values for rgba
        this.hexToRgb(color);
        
        // Calculate end point (extend to map edge)
        this.calculateEndPoint();
    }
    
    hexToRgb(hex) {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        if (result) {
            this.r = parseInt(result[1], 16);
            this.g = parseInt(result[2], 16);
            this.b = parseInt(result[3], 16);
        } else {
            // Default to blue if parsing fails
            this.r = 74;
            this.g = 158;
            this.b = 255;
        }
    }
    
    rgba(alpha) {
        return `rgba(${this.r}, ${this.g}, ${this.b}, ${alpha})`;
    }
    
    calculateEndPoint() {
        // Calculate intersection with map boundaries to extend beam to edge
        // Find the closest intersection point with map boundaries
        let minT = Infinity;
        
        // Check intersections with left and right edges
        if (Math.abs(this.directionX) > 0.001) {
            const tLeft = (0 - this.startX) / this.directionX;
            const tRight = (this.canvasWidth - this.startX) / this.directionX;
            if (tLeft > 0 && isFinite(tLeft)) minT = Math.min(minT, tLeft);
            if (tRight > 0 && isFinite(tRight)) minT = Math.min(minT, tRight);
        }
        
        // Check intersections with top and bottom edges
        if (Math.abs(this.directionY) > 0.001) {
            const tTop = (0 - this.startY) / this.directionY;
            const tBottom = (this.canvasHeight - this.startY) / this.directionY;
            if (tTop > 0 && isFinite(tTop)) minT = Math.min(minT, tTop);
            if (tBottom > 0 && isFinite(tBottom)) minT = Math.min(minT, tBottom);
        }
        
        // Fallback: extend far enough to cover the entire map diagonal
        if (!isFinite(minT) || minT <= 0) {
            minT = Math.hypot(this.canvasWidth, this.canvasHeight);
        }
        
        // Extend slightly beyond the edge for visual effect
        minT *= 1.1;
        
        this.endX = this.startX + this.directionX * minT;
        this.endY = this.startY + this.directionY * minT;
    }
    
    update() {
        this.life -= 1/60; // Assuming 60 FPS
    }
    
    isExpired() {
        return this.life <= 0;
    }
    
    draw(ctx) {
        const alpha = this.life / this.maxLife;
        const pulse = Math.sin(Date.now() / 10) * 0.1 + 0.9; // Pulsing effect
        const currentWidth = this.width * pulse * alpha;
        
        // Outer glow
        ctx.strokeStyle = this.rgba(alpha * 0.4);
        ctx.lineWidth = currentWidth * 2.0;
        ctx.lineCap = 'round';
        ctx.shadowBlur = 20;
        ctx.shadowColor = this.color;
        ctx.beginPath();
        ctx.moveTo(this.startX, this.startY);
        ctx.lineTo(this.endX, this.endY);
        ctx.stroke();
        
        // Middle glow
        ctx.strokeStyle = this.rgba(alpha * 0.6);
        ctx.lineWidth = currentWidth * 1.5;
        ctx.shadowBlur = 15;
        ctx.beginPath();
        ctx.moveTo(this.startX, this.startY);
        ctx.lineTo(this.endX, this.endY);
        ctx.stroke();
        
        // Main beam
        ctx.strokeStyle = this.rgba(alpha * 0.9);
        ctx.lineWidth = currentWidth;
        ctx.shadowBlur = 10;
        ctx.beginPath();
        ctx.moveTo(this.startX, this.startY);
        ctx.lineTo(this.endX, this.endY);
        ctx.stroke();
        
        // Inner bright core
        ctx.strokeStyle = `rgba(255, 255, 255, ${alpha * 0.7})`;
        ctx.lineWidth = currentWidth * 0.4;
        ctx.shadowBlur = 5;
        ctx.beginPath();
        ctx.moveTo(this.startX, this.startY);
        ctx.lineTo(this.endX, this.endY);
        ctx.stroke();
        
        // Reset shadow
        ctx.shadowBlur = 0;
        
        // Draw origin point (where beam starts)
        const originGradient = ctx.createRadialGradient(
            this.startX, this.startY, 0,
            this.startX, this.startY, currentWidth * 0.8
        );
        originGradient.addColorStop(0, `rgba(255, 255, 255, ${alpha * 0.9})`);
        originGradient.addColorStop(0.5, this.rgba(alpha * 0.8));
        originGradient.addColorStop(1, this.rgba(0));
        
        ctx.fillStyle = originGradient;
        ctx.beginPath();
        ctx.arc(this.startX, this.startY, currentWidth * 0.8, 0, Math.PI * 2);
        ctx.fill();
    }
}

// Player Projectile class for player shooting
class PlayerProjectile {
    constructor(x, y, targetX, targetY, speed = 12, damageMultiplier = 1.0, color = '#4a9eff') {
        this.x = x;
        this.y = y;
        this.radius = 6;
        this.speed = speed;
        this.life = 1.0;
        this.maxLife = 3.0; // 3 seconds
        this.color = color;
        this.damageMultiplier = damageMultiplier;
        
        // Calculate direction to target
        const dx = targetX - x;
        const dy = targetY - y;
        const dist = Math.hypot(dx, dy);
        
        if (dist > 0) {
            this.vx = (dx / dist) * speed;
            this.vy = (dy / dist) * speed;
        } else {
            this.vx = speed;
            this.vy = 0;
        }
        
        this.trail = [];
    }
    
    update() {
        this.x += this.vx;
        this.y += this.vy;
        this.life -= 1/60; // Assuming 60 FPS
        
        // Update trail
        this.trail.push({ x: this.x, y: this.y });
        if (this.trail.length > 10) {
            this.trail.shift();
        }
    }
    
    isExpired() {
        return this.life <= 0;
    }
    
    draw(ctx) {
        // Calculate visual enhancements based on charge/damage multiplier
        const sizeMultiplier = 1.0 + (this.damageMultiplier - 1.0) * 0.5; // Scales up to 1.5x at max charge
        const actualRadius = this.radius * sizeMultiplier;
        const chargeProgress = (this.damageMultiplier - 1.0) / 2.0; // 0 to 1
        
        // Enhanced trail with charge-based intensity
        if (this.trail.length > 1) {
            ctx.strokeStyle = this.color + Math.floor(40 + chargeProgress * 40).toString(16).padStart(2, '0');
            ctx.lineWidth = actualRadius * 1.2;
            ctx.lineCap = 'round';
            ctx.shadowBlur = 10 + chargeProgress * 10;
            ctx.shadowColor = this.color;
            ctx.beginPath();
            ctx.moveTo(this.trail[0].x, this.trail[0].y);
            
            this.trail.slice(1).forEach((point, i) => {
                ctx.globalAlpha = (i + 1) / this.trail.length * (0.5 + chargeProgress * 0.3);
                ctx.lineTo(point.x, point.y);
            });
            
            ctx.stroke();
            ctx.globalAlpha = 1;
            ctx.shadowBlur = 0;
        }
        
        // Enhanced glow with charge-based intensity and color shift
        const glowIntensity = Math.min(3.5, 1.5 + this.damageMultiplier * 0.8);
        const glowGradient = ctx.createRadialGradient(this.x, this.y, 0, this.x, this.y, actualRadius * glowIntensity);
        
        // Color shifts from blue to cyan/white as charge increases
        const colorShift = chargeProgress * 0.6;
        const r = Math.min(255, 74 + colorShift * 50);
        const g = Math.min(255, 158 + colorShift * 50);
        const b = Math.min(255, 255);
        const glowColor = `rgba(${r}, ${g}, ${b}, ${0.7 + chargeProgress * 0.3})`;
        
        glowGradient.addColorStop(0, glowColor);
        glowGradient.addColorStop(0.5, this.color + Math.floor(80 + chargeProgress * 40).toString(16).padStart(2, '0'));
        glowGradient.addColorStop(1, this.color + '00');
        
        ctx.fillStyle = glowGradient;
        ctx.shadowBlur = 20 + chargeProgress * 15;
        ctx.shadowColor = this.color;
        ctx.beginPath();
        ctx.arc(this.x, this.y, actualRadius * glowIntensity, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;
        
        // Main projectile body with size scaling
        const bodyGradient = ctx.createRadialGradient(
            this.x - actualRadius/3, 
            this.y - actualRadius/3, 
            0, 
            this.x, 
            this.y, 
            actualRadius
        );
        
        // Inner bright core that gets brighter with charge
        const coreBrightness = 0.6 + chargeProgress * 0.4;
        bodyGradient.addColorStop(0, `rgba(255, 255, 255, ${coreBrightness})`);
        bodyGradient.addColorStop(0.5, this.color + Math.floor(200 + chargeProgress * 55).toString(16).padStart(2, '0'));
        bodyGradient.addColorStop(1, this.color);
        
        ctx.fillStyle = bodyGradient;
        ctx.beginPath();
        ctx.arc(this.x, this.y, actualRadius, 0, Math.PI * 2);
        ctx.fill();
        
        // Enhanced arrowhead with charge-based size and glow
        const angle = Math.atan2(this.vy, this.vx);
        ctx.save();
        ctx.translate(this.x, this.y);
        ctx.rotate(angle);
        
        // Arrowhead glow
        const arrowGlow = ctx.createRadialGradient(0, 0, 0, 0, 0, actualRadius * 2);
        arrowGlow.addColorStop(0, `rgba(255, 255, 255, ${chargeProgress * 0.6})`);
        arrowGlow.addColorStop(1, this.color + '00');
        
        ctx.fillStyle = arrowGlow;
        ctx.beginPath();
        ctx.arc(actualRadius * 0.7, 0, actualRadius * 1.5, 0, Math.PI * 2);
        ctx.fill();
        
        // Arrowhead shape (scales with charge)
        ctx.fillStyle = this.color;
        ctx.shadowBlur = 8 + chargeProgress * 8;
        ctx.shadowColor = this.color;
        ctx.beginPath();
        ctx.moveTo(actualRadius * 1.8, 0);
        ctx.lineTo(-actualRadius * 0.5, -actualRadius * 0.9);
        ctx.lineTo(-actualRadius * 0.3, 0);
        ctx.lineTo(-actualRadius * 0.5, actualRadius * 0.9);
        ctx.closePath();
        ctx.fill();
        
        // Highlight on arrowhead
        ctx.fillStyle = `rgba(255, 255, 255, ${chargeProgress * 0.8})`;
        ctx.beginPath();
        ctx.moveTo(actualRadius * 1.5, 0);
        ctx.lineTo(-actualRadius * 0.2, -actualRadius * 0.5);
        ctx.lineTo(-actualRadius * 0.1, 0);
        ctx.lineTo(-actualRadius * 0.2, actualRadius * 0.5);
        ctx.closePath();
        ctx.fill();
        
        ctx.restore();
        ctx.shadowBlur = 0;
        
        // Pulsing outer ring for highly charged projectiles
        if (chargeProgress > 0.5) {
            const pulse = Math.sin(Date.now() / 50) * 0.3 + 0.7;
            const alphaHex = Math.floor(pulse * chargeProgress * 0.6 * 255).toString(16).padStart(2, '0');
            ctx.strokeStyle = this.color + alphaHex;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(this.x, this.y, actualRadius * (1.3 + pulse * 0.2), 0, Math.PI * 2);
            ctx.stroke();
        }
    }
}

// Projectile class for AI abilities
class Projectile {
    constructor(x, y, targetX, targetY, speed = 8, color = '#ff6b6b') {
        this.x = x;
        this.y = y;
        this.radius = 8;
        this.speed = speed;
        this.life = 1.0;
        this.maxLife = 3.0; // 3 seconds
        this.color = color;
        
        // Calculate direction to target
        const dx = targetX - x;
        const dy = targetY - y;
        const dist = Math.hypot(dx, dy);
        
        this.vx = (dx / dist) * speed;
        this.vy = (dy / dist) * speed;
        
        this.trail = [];
    }
    
    update() {
        this.x += this.vx;
        this.y += this.vy;
        this.life -= 1/60; // Assuming 60 FPS
        
        // Update trail
        this.trail.push({ x: this.x, y: this.y });
        if (this.trail.length > 10) {
            this.trail.shift();
        }
    }
    
    isExpired() {
        return this.life <= 0;
    }
    
    checkCollision(player) {
        return Math.hypot(this.x - player.x, this.y - player.y) < this.radius + player.radius;
    }
    
    draw(ctx) {
        // Draw trail
        if (this.trail.length > 1) {
            ctx.strokeStyle = this.color + '40';
            ctx.lineWidth = this.radius;
            ctx.lineCap = 'round';
            ctx.beginPath();
            ctx.moveTo(this.trail[0].x, this.trail[0].y);
            
            this.trail.slice(1).forEach((point, i) => {
                ctx.globalAlpha = (i + 1) / this.trail.length * 0.5;
                ctx.lineTo(point.x, point.y);
            });
            
            ctx.stroke();
            ctx.globalAlpha = 1;
        }
        
        // Draw glow
        const gradient = ctx.createRadialGradient(this.x, this.y, 0, this.x, this.y, this.radius * 2);
        gradient.addColorStop(0, this.color + '80');
        gradient.addColorStop(1, this.color + '00');
        
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.radius * 2, 0, Math.PI * 2);
        ctx.fill();
        
        // Draw projectile
        ctx.fillStyle = this.color;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
        ctx.fill();
        
        // Inner glow
        const innerGradient = ctx.createRadialGradient(
            this.x - this.radius/3, 
            this.y - this.radius/3, 
            0, 
            this.x, 
            this.y, 
            this.radius
        );
        innerGradient.addColorStop(0, '#ffffff60');
        innerGradient.addColorStop(1, this.color);
        
        ctx.fillStyle = innerGradient;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.radius * 0.8, 0, Math.PI * 2);
        ctx.fill();
    }
}

// Game class
class Game {
    constructor() {
        this.canvas = document.getElementById('gameCanvas');
        this.ctx = this.canvas.getContext('2d');
        
        // Set canvas size first
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
        
        // Game state
        this.isRunning = false;
        this.isPaused = false; // Paused state (game over)
        this.score = 0;
        this.aiAccuracy = 0;
        
        // Timer bar settings
        this.timerMaxDuration = 30; // 30 seconds for bar to fill
        this.gameStartTime = 0;
        
        // Ability unlock thresholds (percentage of bar width)
        // Each tick mark represents unlocking that ability
        // Bar width = (64px + 8px) * 7 - 8px = 496px
        this.abilityUnlockThresholds = [
            { threshold: 64 / 496, slotId: 'dash-ability' },           // W at tick 0
            { threshold: 136 / 496, slotId: 'ability-placeholder-1' },  // E at tick 1
            { threshold: 208 / 496, slotId: 'ability-placeholder-2' },  // R at tick 2
            { threshold: 280 / 496, slotId: 'ability-placeholder-3' },   // T at tick 3
            { threshold: 352 / 496, slotId: 'ability-placeholder-4' },   // Y at tick 4
            { threshold: 424 / 496, slotId: 'ability-placeholder-5' }    // U at tick 5
        ];
        
        // Hover activation state
        this.hoverStartTime = null;
        this.hoverDuration = 1000; // 1 second (50% of original 2 seconds)
        this.isHovering = false;
        
        // Player - spawn bottom left
        this.player = {
            x: 100,
            y: this.canvas.height - 100,
            targetX: 100,
            targetY: this.canvas.height - 100,
            radius: 15,
            color: '#4a9eff',
            trail: [],
            maxSpeed: 4, // Maximum speed cap (reduced by 50%)
            vx: 0, // Velocity for momentum
            vy: 0,
            friction: 0.85, // Friction coefficient for sliding stop (reduced for more sliding)
            isAiming: false,
            isCharging: false,
            chargeStartTime: 0,
            maxChargeTime: 1200, // 1.2 seconds max charge (reduced from 2 seconds)
            aimDirection: { x: 0, y: 0 }
        };
        
        // AI opponent - spawn top right
        this.ai = {
            x: this.canvas.width - 100,
            y: 100,
            vx: 0,
            vy: 0,
            radius: 15,
            color: '#ff4a4a',
            trail: [],
            speed: 1.125, // Reduced by 25% from 1.5
            baseSpeed: 1.125, // Base speed for resetting after boost (reduced by 25%)
            isDashing: false,
            isDashWindup: false,
            dashTime: 0,
            dashWindupTime: 0,
            dashDirection: { x: 0, y: 0 },
            dashStartX: 0,
            dashStartY: 0,
            dashTargetX: 0,
            dashTargetY: 0,
            dashDistanceRemaining: 0,
            isSpeedBoosted: false,
            speedBoostTime: 0,
            isBeamWindup: false,
            beamWindupTime: 0,
            beamDirection: { x: 0, y: 0 }
        };
        
        // AI Abilities
        this.abilities = {
            projectile: {
                cooldown: 3000, // 3 seconds
                lastUsed: 0,
                range: 300
            },
            dash: {
                cooldown: 6000, // 6 seconds
                lastUsed: 0,
                windupDuration: 1500, // 1.5 seconds windup - clear and visible
                dashDistance: 200, // Fixed distance, not speed-based
                dashSpeed: 8 // Slower speed for visibility
            },
            speedBoost: {
                cooldown: 6000, // 6 seconds
                lastUsed: 0,
                duration: 3000, // 3 seconds
                multiplier: 2.0 // Double speed
            },
            slowProjectile: {
                cooldown: 4000, // 4 seconds
                lastUsed: 0,
                range: 300,
                speed: 4, // Slower than regular projectile (8)
                windupDuration: 2000, // 2 seconds cast time (longer than dash)
                beamWidth: 6, // Narrower beam width for easier dodging
                inaccuracyAngle: 0.2 // Random angle offset in radians (~11.5 degrees)
            }
        };
        
        // Projectiles array
        this.projectiles = [];
        
        // Player projectiles array
        this.playerProjectiles = [];
        
        // Player beams array (for full charge attacks)
        this.playerBeams = [];
        
        // AI beams array (for beam ability)
        this.aiBeams = [];
        
        // Neural network for AI (8 inputs: player pos, velocity, AI pos, cooldown states)
        this.nn = new NeuralNetwork(8, 16, 4); // 4 outputs: move_x, move_y, use_projectile, use_dash
        
        // Training data buffer
        this.trainingData = [];
        this.maxTrainingData = 100;
        
        // Particles
        this.particles = [];
        
        // Mouse tracking
        this.mouseX = 100;
        this.mouseY = this.canvas.height - 100;
        
        // Event listeners
        this.setupEventListeners();
        window.addEventListener('resize', () => this.resize());
        
        // Start animation loop
        this.lastTime = 0;
        this.animate(0);
        
        // Initialize abilities - Q unlocked, others locked
        this.lockAllAbilitiesExceptQ();
    }
    
    setupEventListeners() {
        // Mouse movement - always track mouse position
        this.canvas.addEventListener('mousemove', (e) => {
            this.mouseX = e.clientX;
            this.mouseY = e.clientY;
            
            if (this.isRunning && !this.isPaused) {
                this.player.targetX = e.clientX;
                this.player.targetY = e.clientY;
            } else {
                // Update target position even when not running (for cursor display)
                this.player.targetX = e.clientX;
                this.player.targetY = e.clientY;
                
                if (!this.isPaused) {
                    // Check if hovering over player orb (only when not paused)
                    this.checkHover(e.clientX, e.clientY);
                }
            }
        });
        
        // Stop hover when mouse leaves canvas
        this.canvas.addEventListener('mouseleave', () => {
            this.hoverStartTime = null;
            this.isHovering = false;
        });
        
        // Mouse down - start aiming/charging
        this.canvas.addEventListener('mousedown', (e) => {
            if (this.isRunning && !this.isPaused) {
                this.startAiming(e.clientX, e.clientY);
            }
        });
        
        // Mouse up - fire projectile
        this.canvas.addEventListener('mouseup', () => {
            if (this.isRunning && !this.isPaused && this.player.isAiming) {
                this.firePlayerProjectile();
            }
        });
        
        // Spacebar to reset to initial state (hover to restart)
        window.addEventListener('keydown', (e) => {
            if (e.code === 'Space' && this.isPaused) {
                e.preventDefault();
                this.resetToInitialState();
            }
        });
        
        // Restart button click
        document.getElementById('restart-btn').addEventListener('click', () => {
            if (this.isPaused) {
                this.resetToInitialState();
            }
        });
    }
    
    resetToInitialState() {
        // Reset to initial state (not paused, not running) - requires hover to start
        this.isPaused = false;
        this.isRunning = false;
        this.score = 0;
        this.aiAccuracy = 0;
        this.trainingData = [];
        this.gameStartTime = 0;
        
        // Reset timer bar
        const timerBarFill = document.getElementById('timer-bar-fill');
        if (timerBarFill) {
            timerBarFill.style.width = '0%';
        }
        
        // Lock all abilities except Q
        this.lockAllAbilitiesExceptQ();
        
        // Reset hover state
        this.hoverStartTime = null;
        this.isHovering = false;
        
        // Reset positions to original spawn locations
        this.player.x = 100;
        this.player.y = this.canvas.height - 100;
        this.player.targetX = 100;
        this.player.targetY = this.canvas.height - 100;
        this.ai.x = this.canvas.width - 100;
        this.ai.y = 100;
        this.ai.isDashing = false;
        this.ai.isDashWindup = false;
        this.ai.dashTime = 0;
        this.ai.dashWindupTime = 0;
        this.ai.dashDistanceRemaining = 0;
        this.ai.dashStartX = 0;
        this.ai.dashStartY = 0;
        this.ai.dashTargetX = 0;
        this.ai.dashTargetY = 0;
        this.ai.isSpeedBoosted = false;
        this.ai.speedBoostTime = 0;
        this.ai.speed = this.ai.baseSpeed;
        
        // Reset abilities
        this.abilities.projectile.lastUsed = 0;
        this.abilities.dash.lastUsed = 0;
        this.abilities.speedBoost.lastUsed = 0;
        this.abilities.slowProjectile.lastUsed = 0;
        this.ai.isSpeedBoosted = false;
        this.ai.speedBoostTime = 0;
        this.ai.speed = this.ai.baseSpeed;
        this.projectiles = [];
        this.playerProjectiles = [];
        this.playerBeams = [];
        this.aiBeams = [];
        
        // Reset AI beam state
        this.ai.isBeamWindup = false;
        this.ai.beamWindupTime = 0;
        
        // Reset player shooting state
        this.player.isAiming = false;
        this.player.isCharging = false;
        this.player.chargeStartTime = 0;
        this.player.vx = 0;
        this.player.vy = 0;
        
        // Clear particles
        this.particles = [];
        
        // Reset trails
        this.player.trail = [];
        this.ai.trail = [];
        
        // Hide restart button
        document.getElementById('restart-button').classList.add('hidden');
    }
    
    resize() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
        
        // Update positions if not running and objects exist
        if (!this.isRunning && this.player && this.ai) {
            this.player.x = 100;
            this.player.y = this.canvas.height - 100;
            this.player.targetX = 100;
            this.player.targetY = this.canvas.height - 100;
            this.ai.x = this.canvas.width - 100;
            this.ai.y = 100;
            this.mouseX = 100;
            this.mouseY = this.canvas.height - 100;
        }
    }
    
    checkHover(mouseX, mouseY) {
        const dx = mouseX - this.player.x;
        const dy = mouseY - this.player.y;
        const dist = Math.hypot(dx, dy);
        const hoverRadius = this.player.radius + 10; // Slightly larger than orb radius
        
        if (dist < hoverRadius) {
            if (!this.isHovering) {
                this.isHovering = true;
                this.hoverStartTime = Date.now();
            }
            
            // Check if hover duration reached (start immediately when progress reaches 100%)
            const hoverTime = Date.now() - this.hoverStartTime;
            const progress = hoverTime / this.hoverDuration;
            
            if (progress >= 1.0) {
                this.start();
            }
        } else {
            this.isHovering = false;
            this.hoverStartTime = null;
        }
    }
    
    start() {
        this.isRunning = true;
        this.score = 0;
        this.aiAccuracy = 0;
        this.trainingData = [];
        this.gameStartTime = Date.now();
        
        // Reset timer bar
        const timerBarFill = document.getElementById('timer-bar-fill');
        if (timerBarFill) {
            timerBarFill.style.width = '0%';
        }
        
        // Lock all abilities except Q
        this.lockAllAbilitiesExceptQ();
        
        // Reset hover state
        this.hoverStartTime = null;
        this.isHovering = false;
        
        // Reset positions
        this.player.x = 100;
        this.player.y = this.canvas.height - 100;
        this.player.targetX = 100;
        this.player.targetY = this.canvas.height - 100;
        this.ai.x = this.canvas.width - 100;
        this.ai.y = 100;
        this.ai.isDashing = false;
        this.ai.isDashWindup = false;
        this.ai.dashTime = 0;
        this.ai.dashWindupTime = 0;
        this.ai.dashDistanceRemaining = 0;
        this.ai.dashStartX = 0;
        this.ai.dashStartY = 0;
        this.ai.dashTargetX = 0;
        this.ai.dashTargetY = 0;
        this.ai.isSpeedBoosted = false;
        this.ai.speedBoostTime = 0;
        this.ai.speed = this.ai.baseSpeed;
        
        // Reset abilities
        this.abilities.projectile.lastUsed = 0;
        this.abilities.dash.lastUsed = 0;
        this.abilities.speedBoost.lastUsed = 0;
        this.abilities.slowProjectile.lastUsed = 0;
        this.ai.isSpeedBoosted = false;
        this.ai.speedBoostTime = 0;
        this.ai.speed = this.ai.baseSpeed;
        this.projectiles = [];
        this.playerProjectiles = [];
        this.playerBeams = [];
        this.aiBeams = [];
        
        // Reset AI beam state
        this.ai.isBeamWindup = false;
        this.ai.beamWindupTime = 0;
        
        // Reset player shooting state
        this.player.isAiming = false;
        this.player.isCharging = false;
        this.player.chargeStartTime = 0;
        this.player.vx = 0;
        this.player.vy = 0;
        
        // Hide instructions
        document.getElementById('instructions').classList.add('hidden');
    }
    
    startAiming(mouseX, mouseY) {
        this.player.isAiming = true;
        this.player.isCharging = true;
        this.player.chargeStartTime = Date.now();
        
        // Calculate aim direction
        const dx = mouseX - this.player.x;
        const dy = mouseY - this.player.y;
        const dist = Math.hypot(dx, dy);
        
        if (dist > 0) {
            this.player.aimDirection.x = dx / dist;
            this.player.aimDirection.y = dy / dist;
        } else {
            this.player.aimDirection.x = 1;
            this.player.aimDirection.y = 0;
        }
    }
    
    firePlayerProjectile() {
        if (!this.player.isAiming) return;
        
        const chargeTime = Date.now() - this.player.chargeStartTime;
        const chargeProgress = Math.min(1, chargeTime / this.player.maxChargeTime);
        const damageMultiplier = 1.0 + chargeProgress * 2.0; // 1x to 3x damage multiplier
        const isFullCharge = chargeProgress >= 1.0;
        
        // Calculate target position (aim direction * distance)
        const aimDistance = 500; // Distance to aim ahead
        const projectileSpeed = (9 + chargeProgress * 3); // Reduced by 25%: (12 + chargeProgress * 4) * 0.75
        
        if (isFullCharge) {
            // Full charge: shoot a thick beam that extends across the map
            this.playerBeams.push(new PlayerBeam(
                this.player.x,
                this.player.y,
                this.player.aimDirection.x,
                this.player.aimDirection.y,
                this.canvas.width,
                this.canvas.height,
                this.player.color
            ));
            
            // Enhanced muzzle flash for full charge beam
            for (let i = 0; i < 20; i++) {
                this.particles.push(new Particle(
                    this.player.x,
                    this.player.y,
                    this.player.color,
                    {
                        x: this.player.aimDirection.x * 15 + (Math.random() - 0.5) * 12,
                        y: this.player.aimDirection.y * 15 + (Math.random() - 0.5) * 12
                    }
                ));
            }
        } else {
            // Partial charge: shoot 1 bullet straight ahead
            const targetX = this.player.x + this.player.aimDirection.x * aimDistance;
            const targetY = this.player.y + this.player.aimDirection.y * aimDistance;
            
            this.playerProjectiles.push(new PlayerProjectile(
                this.player.x,
                this.player.y,
                targetX,
                targetY,
                projectileSpeed,
                damageMultiplier,
                this.player.color
            ));
            
            // Standard muzzle flash for partial charge
            for (let i = 0; i < 8; i++) {
                this.particles.push(new Particle(
                    this.player.x,
                    this.player.y,
                    this.player.color,
                    {
                        x: this.player.aimDirection.x * 10 + (Math.random() - 0.5) * 8,
                        y: this.player.aimDirection.y * 10 + (Math.random() - 0.5) * 8
                    }
                ));
            }
        }
        
        // Reset aiming state
        this.player.isAiming = false;
        this.player.isCharging = false;
        this.player.chargeStartTime = 0;
    }
    
    updatePlayer() {
        // If aiming/charging, stop movement input but allow momentum
        if (this.player.isAiming) {
            // Check if charge is complete and auto-fire
            const chargeTime = Date.now() - this.player.chargeStartTime;
            const chargeProgress = Math.min(1, chargeTime / this.player.maxChargeTime);
            
            if (chargeProgress >= 1.0) {
                // Auto-fire when charge reaches 100%
                this.firePlayerProjectile();
                return; // Exit early since firePlayerProjectile resets aiming state
            }
            
            // Apply friction to slow down gradually
            this.player.vx *= this.player.friction;
            this.player.vy *= this.player.friction;
            
            // Update aim direction based on current mouse position
            const dx = this.mouseX - this.player.x;
            const dy = this.mouseY - this.player.y;
            const dist = Math.hypot(dx, dy);
            
            if (dist > 0) {
                this.player.aimDirection.x = dx / dist;
                this.player.aimDirection.y = dy / dist;
            }
        } else {
            // Normal movement towards mouse with momentum
            const dx = this.player.targetX - this.player.x;
            const dy = this.player.targetY - this.player.y;
            
            // Apply acceleration towards target
            const accelX = dx * 0.075; // Reduced by 50%
            const accelY = dy * 0.075; // Reduced by 50%
            
            // Update velocity with acceleration
            this.player.vx += accelX;
            this.player.vy += accelY;
            
            // Apply friction for smooth deceleration
            this.player.vx *= this.player.friction;
            this.player.vy *= this.player.friction;
            
            // Cap player speed
            const speed = Math.hypot(this.player.vx, this.player.vy);
            if (speed > this.player.maxSpeed) {
                this.player.vx = (this.player.vx / speed) * this.player.maxSpeed;
                this.player.vy = (this.player.vy / speed) * this.player.maxSpeed;
            }
        }
        
        // Apply velocity to position
        this.player.x += this.player.vx;
        this.player.y += this.player.vy;
        
        // Keep player in bounds
        this.player.x = Math.max(this.player.radius, Math.min(this.canvas.width - this.player.radius, this.player.x));
        this.player.y = Math.max(this.player.radius, Math.min(this.canvas.height - this.player.radius, this.player.y));
        
        // Update trail
        this.player.trail.push({ x: this.player.x, y: this.player.y });
        if (this.player.trail.length > 20) {
            this.player.trail.shift();
        }
        
        // Create particles
        if (Math.random() < 0.3 && !this.player.isAiming) {
            this.particles.push(new Particle(
                this.player.x,
                this.player.y,
                this.player.color,
                {
                    x: (Math.random() - 0.5) * 2,
                    y: (Math.random() - 0.5) * 2
                }
            ));
        }
    }
    
    getNeuralNetworkInputs() {
        const currentTime = Date.now();
        const projectileReady = (currentTime - this.abilities.projectile.lastUsed) >= this.abilities.projectile.cooldown;
        const dashReady = (currentTime - this.abilities.dash.lastUsed) >= this.abilities.dash.cooldown;
        
        return [
            this.player.x / this.canvas.width,
            this.player.y / this.canvas.height,
            this.player.vx / 10,
            this.player.vy / 10,
            this.ai.x / this.canvas.width,
            this.ai.y / this.canvas.height,
            projectileReady ? 1 : 0,
            dashReady ? 1 : 0
        ];
    }
    
    updateAI() {
        const currentTime = Date.now();
        
        // Check ability cooldowns
        const projectileReady = (currentTime - this.abilities.projectile.lastUsed) >= this.abilities.projectile.cooldown;
        const dashReady = (currentTime - this.abilities.dash.lastUsed) >= this.abilities.dash.cooldown;
        const speedBoostReady = (currentTime - this.abilities.speedBoost.lastUsed) >= this.abilities.speedBoost.cooldown;
        const slowProjectileReady = (currentTime - this.abilities.slowProjectile.lastUsed) >= this.abilities.slowProjectile.cooldown;
        
        // Handle speed boost duration
        if (this.ai.isSpeedBoosted) {
            this.ai.speedBoostTime -= 16; // assuming 60fps
            if (this.ai.speedBoostTime <= 0) {
                this.ai.isSpeedBoosted = false;
                this.ai.speed = this.ai.baseSpeed; // Reset to base speed
            }
        }
        
        // Prepare inputs for neural network (normalized)
        const inputs = this.getNeuralNetworkInputs();
        
        // Get AI prediction
        const outputs = this.nn.predict(inputs);
        
        // Convert outputs to actions
        const moveX = (outputs[0] - 0.5) * 2; // -1 to 1
        const moveY = (outputs[1] - 0.5) * 2; // -1 to 1
        const useProjectile = outputs[2] > 0.7; // threshold for using projectile
        const useDash = outputs[3] > 0.8; // threshold for using dash
        
        // Handle beam windup (for slowProjectile ability)
        if (this.ai.isBeamWindup) {
            // Windup phase - AI stops moving, charging up
            this.ai.beamWindupTime -= 16; // assuming 60fps
            
            // Stop AI movement during windup
            this.ai.vx = 0;
            this.ai.vy = 0;
            
            // Lock direction after 50% of windup time to give player time to react
            const windupProgress = 1 - (this.ai.beamWindupTime / this.abilities.slowProjectile.windupDuration);
            const directionLocked = windupProgress >= 0.5;
            
            if (!directionLocked) {
                // Update beam direction to track player (with reduced prediction and less accuracy)
                // Use less prediction time and add inaccuracy
                const predictionTime = 15; // Reduced from 30 for less accuracy
                const futureX = this.player.x + this.player.vx * predictionTime;
                const futureY = this.player.y + this.player.vy * predictionTime;
                const dx = futureX - this.ai.x;
                const dy = futureY - this.ai.y;
                const dist = Math.hypot(dx, dy);
                
                if (dist > 0) {
                    let baseAngle = Math.atan2(dy, dx);
                    // Add random inaccuracy to make it easier to dodge
                    const inaccuracy = (Math.random() - 0.5) * this.abilities.slowProjectile.inaccuracyAngle;
                    baseAngle += inaccuracy;
                    
                    this.ai.beamDirection.x = Math.cos(baseAngle);
                    this.ai.beamDirection.y = Math.sin(baseAngle);
                }
            }
            // After 50% windup, direction is locked - player can see where beam will fire
            
            if (this.ai.beamWindupTime <= 0) {
                // Windup complete, fire beam with narrower width
                this.ai.isBeamWindup = false;
                this.aiBeams.push(new PlayerBeam(
                    this.ai.x,
                    this.ai.y,
                    this.ai.beamDirection.x,
                    this.ai.beamDirection.y,
                    this.canvas.width,
                    this.canvas.height,
                    '#ff99ff', // Pink/magenta color for AI beam
                    this.abilities.slowProjectile.beamWidth // Use narrower width
                ));
                
                // Create muzzle flash effect
                for (let i = 0; i < 20; i++) {
                    this.particles.push(new Particle(
                        this.ai.x,
                        this.ai.y,
                        '#ff99ff',
                        {
                            x: this.ai.beamDirection.x * 15 + (Math.random() - 0.5) * 12,
                            y: this.ai.beamDirection.y * 15 + (Math.random() - 0.5) * 12
                        }
                    ));
                }
            }
        }
        
        // Handle dash ability (windup and execution)
        if (this.ai.isDashWindup) {
            // Windup phase - AI stops moving, charging up
            this.ai.dashWindupTime -= 16; // assuming 60fps
            
            // Stop AI movement during windup for clarity
            this.ai.vx = 0;
            this.ai.vy = 0;
            
            if (this.ai.dashWindupTime <= 0) {
                // Windup complete, start actual dash
                this.ai.isDashWindup = false;
                this.ai.isDashing = true;
                this.ai.dashStartX = this.ai.x;
                this.ai.dashStartY = this.ai.y;
                this.ai.dashDistanceRemaining = this.abilities.dash.dashDistance;
                
                // Calculate target position
                this.ai.dashTargetX = this.ai.x + this.ai.dashDirection.x * this.abilities.dash.dashDistance;
                this.ai.dashTargetY = this.ai.y + this.ai.dashDirection.y * this.abilities.dash.dashDistance;
                
                // Set velocity towards target
                this.ai.vx = this.ai.dashDirection.x * this.abilities.dash.dashSpeed;
                this.ai.vy = this.ai.dashDirection.y * this.abilities.dash.dashSpeed;
            }
        } else if (this.ai.isDashing) {
            // Dash execution phase - move towards target
            const dx = this.ai.dashTargetX - this.ai.x;
            const dy = this.ai.dashTargetY - this.ai.y;
            const distanceToTarget = Math.hypot(dx, dy);
            
            // Check if we've reached or passed the target
            if (distanceToTarget < 5 || this.ai.dashDistanceRemaining <= 0) {
                // Dash complete - snap to target if close
                this.ai.x = this.ai.dashTargetX;
                this.ai.y = this.ai.dashTargetY;
                this.ai.isDashing = false;
                this.ai.vx = 0;
                this.ai.vy = 0;
            } else {
                // Continue dashing
                this.ai.dashDistanceRemaining -= Math.hypot(this.ai.vx, this.ai.vy);
                
                // Create dash particles
                for (let i = 0; i < 5; i++) {
                    this.particles.push(new Particle(
                        this.ai.x + (Math.random() - 0.5) * 20,
                        this.ai.y + (Math.random() - 0.5) * 20,
                        '#ffaa00',
                        {
                            x: (Math.random() - 0.5) * 6,
                            y: (Math.random() - 0.5) * 6
                        }
                    ));
                }
            }
        } else {
            // Normal movement (speed affected by speed boost)
            const currentSpeed = this.ai.speed;
            this.ai.vx = moveX * currentSpeed;
            this.ai.vy = moveY * currentSpeed;
            
            // Use abilities based on neural network decision (only if unlocked)
            const distToPlayer = Math.hypot(this.player.x - this.ai.x, this.player.y - this.ai.y);
            
            if (useProjectile && projectileReady && this.isAbilityUnlocked('projectile')) {
                this.useProjectile();
            }
            
            if (useDash && dashReady && !this.ai.isDashing && !this.ai.isDashWindup && this.isAbilityUnlocked('dash')) {
                this.useDash(moveX, moveY);
            }
            
            // Use speed boost if ready and unlocked (simple logic - use when player is far)
            if (speedBoostReady && !this.ai.isSpeedBoosted && distToPlayer > 200 && this.isAbilityUnlocked('speedBoost')) {
                this.useSpeedBoost();
            }
            
            // Use slow projectile (beam) if ready and unlocked (when regular projectile is on cooldown)
            if (slowProjectileReady && !projectileReady && !this.ai.isBeamWindup && distToPlayer < this.abilities.slowProjectile.range && this.isAbilityUnlocked('slowProjectile')) {
                this.useSlowProjectile();
            }
        }
        
        this.ai.x += this.ai.vx;
        this.ai.y += this.ai.vy;
        
        // Keep AI in bounds
        this.ai.x = Math.max(this.ai.radius, Math.min(this.canvas.width - this.ai.radius, this.ai.x));
        this.ai.y = Math.max(this.ai.radius, Math.min(this.canvas.height - this.ai.radius, this.ai.y));
        
        // Update trail
        this.ai.trail.push({ x: this.ai.x, y: this.ai.y });
        if (this.ai.trail.length > 20) {
            this.ai.trail.shift();
        }
        
        // Create particles
        if (Math.random() < 0.3) {
            this.particles.push(new Particle(
                this.ai.x,
                this.ai.y,
                this.ai.color,
                {
                    x: (Math.random() - 0.5) * 2,
                    y: (Math.random() - 0.5) * 2
                }
            ));
        }
        
        // Collect training data
        const futurePlayerX = this.player.x + this.player.vx * 15;
        const futurePlayerY = this.player.y + this.player.vy * 15;
        const distToPlayer = Math.hypot(this.player.x - this.ai.x, this.player.y - this.ai.y);
        
        // Calculate optimal actions for training
        const optimalMoveX = Math.min(1, Math.max(-1, (futurePlayerX - this.ai.x) / 100));
        const optimalMoveY = Math.min(1, Math.max(-1, (futurePlayerY - this.ai.y) / 100));
        const shouldUseProjectile = distToPlayer < this.abilities.projectile.range && distToPlayer > 50 ? 0.9 : 0.1;
        const shouldUseDash = distToPlayer > 150 && distToPlayer < 250 ? 0.9 : 0.1;
        
        this.trainingData.push({
            inputs: inputs,
            targets: [
                (optimalMoveX + 1) / 2, // normalize to 0-1
                (optimalMoveY + 1) / 2, // normalize to 0-1
                shouldUseProjectile,
                shouldUseDash
            ]
        });
        
        if (this.trainingData.length > this.maxTrainingData) {
            this.trainingData.shift();
        }
        
        // Train the network periodically
        if (this.trainingData.length > 15 && Math.random() < 0.08) {
            const sampleSize = Math.min(8, this.trainingData.length);
            const totalError = Array.from({ length: sampleSize }, (_, i) => {
                const data = this.trainingData[this.trainingData.length - 1 - i];
                return this.nn.train(data.inputs, data.targets);
            }).reduce((sum, err) => sum + err, 0);
            
            this.aiAccuracy = Math.max(0, Math.min(100, 100 - (totalError / sampleSize) * 150));
        }
    }
    
    useProjectile() {
        const currentTime = Date.now();
        this.abilities.projectile.lastUsed = currentTime;
        
        // Predict where player will be
        const futureX = this.player.x + this.player.vx * 20;
        const futureY = this.player.y + this.player.vy * 20;
        
        this.projectiles.push(new Projectile(this.ai.x, this.ai.y, futureX, futureY));
        
        // Create muzzle flash effect
        for (let i = 0; i < 8; i++) {
            this.particles.push(new Particle(
                this.ai.x,
                this.ai.y,
                '#ff6b6b',
                {
                    x: (Math.random() - 0.5) * 10,
                    y: (Math.random() - 0.5) * 10
                }
            ));
        }
    }
    
    useDash(dirX, dirY) {
        const currentTime = Date.now();
        this.abilities.dash.lastUsed = currentTime;
        
        // Start windup phase
        this.ai.isDashWindup = true;
        this.ai.dashWindupTime = this.abilities.dash.windupDuration;
        
        // Store windup start position
        this.ai.dashStartX = this.ai.x;
        this.ai.dashStartY = this.ai.y;
        
        // Normalize direction
        const length = Math.hypot(dirX, dirY);
        if (length > 0) {
            this.ai.dashDirection.x = dirX / length;
            this.ai.dashDirection.y = dirY / length;
        } else {
            // Default direction towards player
            const dx = this.player.x - this.ai.x;
            const dy = this.player.y - this.ai.y;
            const dist = Math.hypot(dx, dy);
            this.ai.dashDirection.x = dx / dist;
            this.ai.dashDirection.y = dy / dist;
        }
        
        // Pre-calculate target position for visual clarity
        this.ai.dashTargetX = this.ai.x + this.ai.dashDirection.x * this.abilities.dash.dashDistance;
        this.ai.dashTargetY = this.ai.y + this.ai.dashDirection.y * this.abilities.dash.dashDistance;
    }
    
    useSpeedBoost() {
        const currentTime = Date.now();
        this.abilities.speedBoost.lastUsed = currentTime;
        this.ai.isSpeedBoosted = true;
        this.ai.speedBoostTime = this.abilities.speedBoost.duration;
        this.ai.speed = this.ai.baseSpeed * this.abilities.speedBoost.multiplier;
        
        // Create speed boost particles
        for (let i = 0; i < 10; i++) {
            this.particles.push(new Particle(
                this.ai.x + (Math.random() - 0.5) * 30,
                this.ai.y + (Math.random() - 0.5) * 30,
                '#00ff88',
                {
                    x: (Math.random() - 0.5) * 4,
                    y: (Math.random() - 0.5) * 4
                }
            ));
        }
    }
    
    useSlowProjectile() {
        const currentTime = Date.now();
        this.abilities.slowProjectile.lastUsed = currentTime;
        
        // Start beam windup phase
        this.ai.isBeamWindup = true;
        this.ai.beamWindupTime = this.abilities.slowProjectile.windupDuration;
        
        // Calculate direction towards player (with reduced prediction and inaccuracy)
        const predictionTime = 15; // Reduced prediction for less accuracy
        const futureX = this.player.x + this.player.vx * predictionTime;
        const futureY = this.player.y + this.player.vy * predictionTime;
        const dx = futureX - this.ai.x;
        const dy = futureY - this.ai.y;
        const dist = Math.hypot(dx, dy);
        
        if (dist > 0) {
            let baseAngle = Math.atan2(dy, dx);
            // Add random inaccuracy to initial direction
            const inaccuracy = (Math.random() - 0.5) * this.abilities.slowProjectile.inaccuracyAngle;
            baseAngle += inaccuracy;
            
            this.ai.beamDirection.x = Math.cos(baseAngle);
            this.ai.beamDirection.y = Math.sin(baseAngle);
        } else {
            this.ai.beamDirection.x = 1;
            this.ai.beamDirection.y = 0;
        }
    }
    
    updateProjectiles() {
        for (let i = this.projectiles.length - 1; i >= 0; i--) {
            const projectile = this.projectiles[i];
            projectile.update();
            
            // Check collision with player
            if (projectile.checkCollision(this.player)) {
                this.gameOver('Hit by projectile!');
                return;
            }
            
            // Check if projectile is out of bounds or expired
            if (projectile.isExpired() || 
                projectile.x < 0 || projectile.x > this.canvas.width ||
                projectile.y < 0 || projectile.y > this.canvas.height) {
                this.projectiles.splice(i, 1);
            }
        }
        
        // Update player projectiles
        for (let i = this.playerProjectiles.length - 1; i >= 0; i--) {
            const projectile = this.playerProjectiles[i];
            projectile.update();
            
            // Check if projectile is out of bounds or expired
            // Note: Player projectiles do NOT damage the red orb (AI) as per requirements
            if (projectile.isExpired() || 
                projectile.x < 0 || projectile.x > this.canvas.width ||
                projectile.y < 0 || projectile.y > this.canvas.height) {
                this.playerProjectiles.splice(i, 1);
            }
        }
        
        // Update player beams
        for (let i = this.playerBeams.length - 1; i >= 0; i--) {
            const beam = this.playerBeams[i];
            beam.update();
            
            // Remove expired beams
            if (beam.isExpired()) {
                this.playerBeams.splice(i, 1);
            }
        }
        
        // Update AI beams
        for (let i = this.aiBeams.length - 1; i >= 0; i--) {
            const beam = this.aiBeams[i];
            beam.update();
            
            // Check collision with player
            // Beam collision check - simple line segment collision
            const beamLength = Math.hypot(beam.endX - beam.startX, beam.endY - beam.startY);
            const beamDirX = (beam.endX - beam.startX) / beamLength;
            const beamDirY = (beam.endY - beam.startY) / beamLength;
            
            // Project player position onto beam line
            const toPlayerX = this.player.x - beam.startX;
            const toPlayerY = this.player.y - beam.startY;
            const projection = toPlayerX * beamDirX + toPlayerY * beamDirY;
            
            if (projection >= 0 && projection <= beamLength) {
                // Check distance from beam line to player
                const closestX = beam.startX + beamDirX * projection;
                const closestY = beam.startY + beamDirY * projection;
                const distToBeam = Math.hypot(this.player.x - closestX, this.player.y - closestY);
                
                if (distToBeam < beam.width / 2 + this.player.radius * 0.8) { // Slightly more forgiving collision
                    this.gameOver('Hit by beam!');
                    return;
                }
            }
            
            // Remove expired beams
            if (beam.isExpired()) {
                this.aiBeams.splice(i, 1);
            }
        }
    }
    
    checkCollision() {
        const dx = this.player.x - this.ai.x;
        const dy = this.player.y - this.ai.y;
        if (Math.hypot(dx, dy) < this.player.radius + this.ai.radius) {
            this.gameOver('Caught by AI!');
        }
    }
    
    gameOver(reason) {
        this.isPaused = true;
        this.isRunning = false;
        
        // Show restart button
        document.getElementById('restart-button').classList.remove('hidden');
    }
    
    start() {
        this.isRunning = true;
        this.isPaused = false;
        this.score = 0;
        this.aiAccuracy = 0;
        this.trainingData = [];
        this.gameStartTime = Date.now();
        
        // Reset timer bar
        const timerBarFill = document.getElementById('timer-bar-fill');
        if (timerBarFill) {
            timerBarFill.style.width = '0%';
        }
        
        // Lock all abilities except Q
        this.lockAllAbilitiesExceptQ();
        
        // Reset hover state
        this.hoverStartTime = null;
        this.isHovering = false;
        
        // Reset positions to original spawn locations
        this.player.x = 100;
        this.player.y = this.canvas.height - 100;
        this.player.targetX = 100;
        this.player.targetY = this.canvas.height - 100;
        this.ai.x = this.canvas.width - 100;
        this.ai.y = 100;
        this.ai.isDashing = false;
        this.ai.isDashWindup = false;
        this.ai.isBeamWindup = false;
        this.ai.dashTime = 0;
        this.ai.dashWindupTime = 0;
        this.ai.beamWindupTime = 0;
        this.ai.dashDistanceRemaining = 0;
        this.ai.dashStartX = 0;
        this.ai.dashStartY = 0;
        this.ai.dashTargetX = 0;
        this.ai.dashTargetY = 0;
        this.ai.isSpeedBoosted = false;
        this.ai.speedBoostTime = 0;
        this.ai.speed = this.ai.baseSpeed;
        
        // Reset abilities
        this.abilities.projectile.lastUsed = 0;
        this.abilities.dash.lastUsed = 0;
        this.abilities.speedBoost.lastUsed = 0;
        this.abilities.slowProjectile.lastUsed = 0;
        this.ai.isSpeedBoosted = false;
        this.ai.speedBoostTime = 0;
        this.ai.speed = this.ai.baseSpeed;
        this.projectiles = [];
        
        // Clear particles
        this.particles = [];
        
        // Reset trails
        this.player.trail = [];
        this.ai.trail = [];
        
        // Hide restart button
        document.getElementById('restart-button').classList.add('hidden');
    }
    
    updateAbilityUI() {
        const currentTime = Date.now();
        
        const updateAbilitySlot = (abilityKey, slotId) => {
            const slot = document.getElementById(slotId);
            if (!slot || slot.classList.contains('locked')) {
                return; // Skip locked abilities
            }
            
            const ability = this.abilities[abilityKey];
            const cooldownLeft = Math.max(0, ability.cooldown - (currentTime - ability.lastUsed));
            const progress = (cooldownLeft / ability.cooldown) * 100;
            
            const cooldownOverlay = slot.querySelector('.cooldown-overlay');
            const cooldownTimer = slot.querySelector('.cooldown-timer');
            
            if (cooldownLeft > 0) {
                slot.classList.add('cooldown');
                slot.classList.remove('active');
                if (cooldownOverlay) {
                    cooldownOverlay.style.setProperty('--progress', `${progress}%`);
                }
                if (cooldownTimer) {
                    cooldownTimer.textContent = Math.ceil(cooldownLeft / 1000);
                }
            } else {
                slot.classList.remove('cooldown');
                slot.classList.add('active');
                if (cooldownTimer) {
                    cooldownTimer.textContent = '';
                }
            }
        };
        
        updateAbilitySlot('projectile', 'projectile-ability');
        updateAbilitySlot('dash', 'dash-ability');
        updateAbilitySlot('speedBoost', 'ability-placeholder-1'); // E
        updateAbilitySlot('slowProjectile', 'ability-placeholder-2'); // R
    }
    
    isAbilityUnlocked(abilityKey) {
        // Q (projectile) is always unlocked
        if (abilityKey === 'projectile') {
            return true;
        }
        
        // Check unlock thresholds
        const progress = Math.min(1, this.score / this.timerMaxDuration);
        
        // Map ability keys to their unlock thresholds
        const unlockMap = {
            'dash': 64 / 496,           // W at tick 0
            'speedBoost': 136 / 496,     // E at tick 1
            'slowProjectile': 208 / 496  // R at tick 2
        };
        
        const threshold = unlockMap[abilityKey];
        return threshold !== undefined && progress >= threshold;
    }
    
    updateTimerBar() {
        if (!this.isRunning || this.isPaused) {
            return;
        }
        
        const progress = Math.min(1, this.score / this.timerMaxDuration); // 0 to 1
        const progressPercent = progress * 100; // 0 to 100
        
        const timerBarFill = document.getElementById('timer-bar-fill');
        if (timerBarFill) {
            timerBarFill.style.width = `${progressPercent}%`;
        }
        
        // Unlock abilities as the bar reaches each tick mark threshold
        this.abilityUnlockThresholds.forEach(({ threshold, slotId }) => {
            if (progress >= threshold) {
                this.unlockAbility(slotId);
            }
        });
    }
    
    unlockAbility(slotId) {
        const slot = document.getElementById(slotId);
        if (slot && slot.classList.contains('locked')) {
            slot.classList.remove('locked');
            
            // Remove lock overlay
            const lockOverlay = slot.querySelector('.lock-overlay');
            if (lockOverlay) {
                lockOverlay.remove();
            }
            
            // Update icon if it's a placeholder
            const icon = slot.querySelector('.ability-icon');
            if (icon && icon.textContent === '🔒') {
                // Set appropriate icon based on slot
                const keybind = slot.querySelector('.ability-keybind')?.textContent;
                if (keybind === 'W') {
                    icon.textContent = '⚡';
                } else if (keybind === 'E') {
                    icon.textContent = '🚀'; // Speed boost icon
                } else if (keybind === 'R') {
                    icon.textContent = '💫'; // Slow projectile icon
                } else {
                    // Placeholder icons for other abilities
                    icon.textContent = '✨';
                }
            }
            
            // Add active class
            slot.classList.add('active');
        }
    }
    
    lockAllAbilitiesExceptQ() {
        // Lock all abilities except Q (projectile-ability)
        const abilitiesToLock = [
            'dash-ability',
            'ability-placeholder-1',
            'ability-placeholder-2',
            'ability-placeholder-3',
            'ability-placeholder-4',
            'ability-placeholder-5'
        ];
        
        abilitiesToLock.forEach(slotId => {
            const slot = document.getElementById(slotId);
            if (slot) {
                slot.classList.add('locked');
                slot.classList.remove('active');
                
                // Add lock overlay if it doesn't exist
                if (!slot.querySelector('.lock-overlay')) {
                    const lockOverlay = document.createElement('div');
                    lockOverlay.className = 'lock-overlay';
                    slot.appendChild(lockOverlay);
                }
                
                // Update icon for placeholders
                const icon = slot.querySelector('.ability-icon');
                const keybind = slot.querySelector('.ability-keybind')?.textContent;
                if (icon && keybind === 'W') {
                    icon.textContent = '⚡';
                } else if (icon && keybind === 'E') {
                    icon.textContent = '🚀';
                } else if (icon && keybind === 'R') {
                    icon.textContent = '💫';
                } else if (icon) {
                    icon.textContent = '🔒';
                }
            }
        });
    }
    
    updateParticles() {
        for (let i = this.particles.length - 1; i >= 0; i--) {
            const particle = this.particles[i];
            particle.update();
            
            if (particle.life <= 0) {
                this.particles.splice(i, 1);
            }
        }
    }
    
    update(deltaTime) {
        if (!this.isRunning || this.isPaused) {
            // Check hover activation progress even when not running
            if (!this.isRunning && !this.isPaused && this.isHovering && this.hoverStartTime) {
                const hoverTime = Date.now() - this.hoverStartTime;
                const progress = hoverTime / this.hoverDuration;
                
                if (progress >= 1.0) {
                    this.start();
                }
            }
            return;
        }
        
        this.updatePlayer();
        this.updateAI();
        this.updateProjectiles();
        this.checkCollision();
        this.updateParticles();
        this.updateAbilityUI();
        
        // Update score
        this.score += deltaTime;
        
        // Update timer bar
        this.updateTimerBar();
    }
    
    addAlphaToColor(color, alpha) {
        // Convert alpha (0-1) to hex (00-FF)
        const alphaHex = Math.floor(alpha * 255).toString(16).padStart(2, '0');
        
        // If color is already hex format, append alpha
        if (color.startsWith('#')) {
            return color + alphaHex;
        }
        
        // If color is RGB format, convert to RGBA
        if (color.startsWith('rgb(')) {
            return color.replace('rgb(', 'rgba(').replace(')', `, ${alpha})`);
        }
        
        return color;
    }
    
    drawTrail(ctx, trail, color, lineWidth) {
        if (trail.length < 2) return;
        
        ctx.strokeStyle = this.addAlphaToColor(color, 0.25);
        ctx.lineWidth = lineWidth;
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(trail[0].x, trail[0].y);
        
        trail.slice(1).forEach((point, i) => {
            ctx.globalAlpha = (i + 1) / trail.length * 0.3;
            ctx.lineTo(point.x, point.y);
        });
        
        ctx.stroke();
        ctx.globalAlpha = 1;
    }
    
    drawGlow(ctx, x, y, radius, color, intensity = 3) {
        const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius * intensity);
        gradient.addColorStop(0, this.addAlphaToColor(color, 0.25));
        gradient.addColorStop(1, this.addAlphaToColor(color, 0));
        
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(x, y, radius * intensity, 0, Math.PI * 2);
        ctx.fill();
    }
    
    drawOrb(x, y, radius, color, trail, activationProgress = 0) {
        // Desaturate colors when paused (like League of Legends death screen)
        const saturation = this.isPaused ? 0.4 : 1.0;
        const desaturatedColor = this.desaturateColor(color, saturation);
        
        this.drawTrail(this.ctx, trail, desaturatedColor, radius * 2);
        
        // Draw activation ring if hovering
        if (activationProgress > 0) {
            const ringRadius = radius + 15 + activationProgress * 5;
            const ringAlpha = activationProgress * 0.8;
            
            this.ctx.strokeStyle = this.addAlphaToColor(desaturatedColor, ringAlpha);
            this.ctx.lineWidth = 3;
            this.ctx.beginPath();
            this.ctx.arc(x, y, ringRadius, 0, Math.PI * 2);
            this.ctx.stroke();
            
            // Draw progress arc
            const progressAngle = activationProgress * Math.PI * 2;
            this.ctx.strokeStyle = desaturatedColor;
            this.ctx.lineWidth = 4;
            this.ctx.lineCap = 'round';
            this.ctx.beginPath();
            this.ctx.arc(x, y, radius + 20, -Math.PI / 2, -Math.PI / 2 + progressAngle);
            this.ctx.stroke();
        }
        
        this.drawGlow(this.ctx, x, y, radius, desaturatedColor, activationProgress > 0 ? 4 + activationProgress * 2 : 3);
        
        // Draw orb
        this.ctx.fillStyle = desaturatedColor;
        this.ctx.beginPath();
        this.ctx.arc(x, y, radius, 0, Math.PI * 2);
        this.ctx.fill();
        
        // Inner glow
        const innerGradient = this.ctx.createRadialGradient(x - radius/3, y - radius/3, 0, x, y, radius);
        innerGradient.addColorStop(0, this.addAlphaToColor('#ffffff', 0.25));
        innerGradient.addColorStop(1, desaturatedColor);
        
        this.ctx.fillStyle = innerGradient;
        this.ctx.beginPath();
        this.ctx.arc(x, y, radius * 0.9, 0, Math.PI * 2);
        this.ctx.fill();
    }
    
    drawHoverRadius() {
        if (this.isRunning || this.isPaused) return;
        
        const hoverRadius = this.player.radius + 10;
        const pulse = Math.sin(Date.now() / 500) * 0.15 + 0.85; // Pulse between 0.7 and 1.0
        
        // Draw outer glow
        const gradient = this.ctx.createRadialGradient(
            this.player.x, this.player.y, hoverRadius - 5,
            this.player.x, this.player.y, hoverRadius + 15
        );
        gradient.addColorStop(0, this.player.color + '40');
        gradient.addColorStop(1, this.player.color + '00');
        
        this.ctx.fillStyle = gradient;
        this.ctx.beginPath();
        this.ctx.arc(this.player.x, this.player.y, hoverRadius + 15 * pulse, 0, Math.PI * 2);
        this.ctx.fill();
        
        // Draw dashed circle outline
        this.ctx.strokeStyle = this.player.color + Math.floor(150 * pulse).toString(16).padStart(2, '0');
        this.ctx.lineWidth = 2;
        this.ctx.setLineDash([8, 4]);
        this.ctx.beginPath();
        this.ctx.arc(this.player.x, this.player.y, hoverRadius, 0, Math.PI * 2);
        this.ctx.stroke();
        this.ctx.setLineDash([]);
        
        // Draw inner ring
        this.ctx.strokeStyle = this.player.color + Math.floor(100 * pulse).toString(16).padStart(2, '0');
        this.ctx.lineWidth = 1;
        this.ctx.beginPath();
        this.ctx.arc(this.player.x, this.player.y, hoverRadius - 3, 0, Math.PI * 2);
        this.ctx.stroke();
    }
    
    hexToRgb(hex) {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? {
            r: parseInt(result[1], 16),
            g: parseInt(result[2], 16),
            b: parseInt(result[3], 16)
        } : null;
    }
    
    desaturateColor(color, saturation) {
        const rgb = this.hexToRgb(color);
        if (!rgb) return color;
        
        // Convert to grayscale
        const gray = rgb.r * 0.299 + rgb.g * 0.587 + rgb.b * 0.114;
        
        // Interpolate between original and grayscale
        const r = Math.round(rgb.r * saturation + gray * (1 - saturation));
        const g = Math.round(rgb.g * saturation + gray * (1 - saturation));
        const b = Math.round(rgb.b * saturation + gray * (1 - saturation));
        
        return `rgb(${r}, ${g}, ${b})`;
    }
    
    drawDeathOverlay() {
        // Dark overlay like League of Legends death screen (reduced opacity)
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.4)';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    }
    
    drawAimingIndicator() {
        if (!this.player.isAiming) return;
        
        const chargeTime = Date.now() - this.player.chargeStartTime;
        const chargeProgress = Math.min(1, chargeTime / this.player.maxChargeTime);
        const damageMultiplier = 1.0 + chargeProgress * 2.0;
        
        // Create charge particles for visual feedback
        if (Math.random() < 0.3) {
            const angle = Math.random() * Math.PI * 2;
            const dist = this.player.radius + Math.random() * 30;
            this.particles.push(new Particle(
                this.player.x + Math.cos(angle) * dist,
                this.player.y + Math.sin(angle) * dist,
                this.player.color,
                {
                    x: Math.cos(angle) * (2 + chargeProgress * 3),
                    y: Math.sin(angle) * (2 + chargeProgress * 3)
                }
            ));
        }
        
        // Draw aim line with enhanced visuals
        const aimDistance = 500;
        const aimEndX = this.player.x + this.player.aimDirection.x * aimDistance;
        const aimEndY = this.player.y + this.player.aimDirection.y * aimDistance;
        
        // Main aim line with pulsing effect
        const pulse = Math.sin(chargeProgress * Math.PI * 10) * 0.15 + 0.85;
        const lineAlpha = 0.4 + chargeProgress * 0.6;
        const lineWidth = 2 + chargeProgress * 4;
        
        // Glow effect along the line
        const gradient = this.ctx.createLinearGradient(
            this.player.x, this.player.y,
            aimEndX, aimEndY
        );
        const colorIntensity = 0.3 + chargeProgress * 0.7;
        gradient.addColorStop(0, this.addAlphaToColor(this.player.color, lineAlpha * 0.8));
        gradient.addColorStop(0.5, this.addAlphaToColor(this.player.color, lineAlpha));
        gradient.addColorStop(1, this.addAlphaToColor(this.player.color, lineAlpha * 0.9));
        
        this.ctx.strokeStyle = gradient;
        this.ctx.lineWidth = lineWidth * pulse;
        this.ctx.lineCap = 'round';
        this.ctx.shadowBlur = 15 + chargeProgress * 15;
        this.ctx.shadowColor = this.player.color;
        this.ctx.beginPath();
        this.ctx.moveTo(this.player.x, this.player.y);
        this.ctx.lineTo(aimEndX, aimEndY);
        this.ctx.stroke();
        this.ctx.shadowBlur = 0;
        
        // Draw multiple expanding charge rings around player with color shift
        const ringCount = 3;
        for (let i = 0; i < ringCount; i++) {
            const ringProgress = chargeProgress + (i * 0.15);
            const ringRadius = this.player.radius + ringProgress * 25;
            const ringAlpha = (0.15 + chargeProgress * 0.5) * (1 - i * 0.3);
            
            // Color shifts from blue to cyan/white as charge increases
            const colorShift = chargeProgress * 0.5;
            const r = Math.min(255, 74 + colorShift * 50);
            const g = Math.min(255, 158 + colorShift * 50);
            const b = Math.min(255, 255);
            const ringColor = `rgba(${r}, ${g}, ${b}, ${ringAlpha})`;
            
            this.ctx.strokeStyle = ringColor;
            this.ctx.lineWidth = 2 + chargeProgress * 2;
            this.ctx.beginPath();
            this.ctx.arc(this.player.x, this.player.y, ringRadius, 0, Math.PI * 2);
            this.ctx.stroke();
        }
        
        // Draw pulsing inner ring with faster pulse at higher charge
        const innerPulse = Math.sin(chargeProgress * Math.PI * 12) * 0.3 + 0.7;
        const innerRingRadius = this.player.radius + chargeProgress * 15;
        const innerColor = `rgba(255, 255, 255, ${innerPulse * chargeProgress * 0.8})`;
        
        this.ctx.strokeStyle = innerColor;
        this.ctx.lineWidth = 3 + chargeProgress * 3;
        this.ctx.beginPath();
        this.ctx.arc(this.player.x, this.player.y, innerRingRadius, 0, Math.PI * 2);
        this.ctx.stroke();
        
        // Enhanced charge progress arc with glow
        const arcRadius = this.player.radius + 20;
        const startAngle = -Math.PI / 2;
        const endAngle = startAngle + chargeProgress * Math.PI * 2;
        
        // Outer glow arc
        this.ctx.strokeStyle = this.addAlphaToColor(this.player.color, 0.3);
        this.ctx.lineWidth = 8;
        this.ctx.lineCap = 'round';
        this.ctx.shadowBlur = 20;
        this.ctx.shadowColor = this.player.color;
        this.ctx.beginPath();
        this.ctx.arc(this.player.x, this.player.y, arcRadius, startAngle, endAngle);
        this.ctx.stroke();
        
        // Main progress arc
        this.ctx.strokeStyle = this.player.color;
        this.ctx.lineWidth = 5;
        this.ctx.shadowBlur = 15;
        this.ctx.beginPath();
        this.ctx.arc(this.player.x, this.player.y, arcRadius, startAngle, endAngle);
        this.ctx.stroke();
        
        // Inner bright arc
        const brightColor = `rgba(255, 255, 255, ${chargeProgress * 0.9})`;
        this.ctx.strokeStyle = brightColor;
        this.ctx.lineWidth = 3;
        this.ctx.shadowBlur = 10;
        this.ctx.beginPath();
        this.ctx.arc(this.player.x, this.player.y, arcRadius, startAngle, endAngle);
        this.ctx.stroke();
        
        this.ctx.shadowBlur = 0;
        
        // Enhanced aim point indicator with pulsing effect
        const pointRadius = 5 + chargeProgress * 8;
        const pointPulse = Math.sin(chargeProgress * Math.PI * 8) * 0.2 + 0.8;
        
        // Outer glow rings
        for (let i = 0; i < 3; i++) {
            const ringRad = pointRadius * (1.5 + i * 0.8);
            const ringAlpha = (0.3 - i * 0.1) * chargeProgress * pointPulse;
            const pointGradient = this.ctx.createRadialGradient(
                aimEndX, aimEndY, 0,
                aimEndX, aimEndY, ringRad
            );
            pointGradient.addColorStop(0, this.addAlphaToColor(this.player.color, ringAlpha));
            pointGradient.addColorStop(1, this.addAlphaToColor(this.player.color, 0));
            
            this.ctx.fillStyle = pointGradient;
            this.ctx.beginPath();
            this.ctx.arc(aimEndX, aimEndY, ringRad, 0, Math.PI * 2);
            this.ctx.fill();
        }
        
        // Main aim point
        const mainGradient = this.ctx.createRadialGradient(
            aimEndX, aimEndY, 0,
            aimEndX, aimEndY, pointRadius * pointPulse
        );
        mainGradient.addColorStop(0, `rgba(255, 255, 255, ${chargeProgress * 0.9})`);
        mainGradient.addColorStop(0.5, this.addAlphaToColor(this.player.color, 0.8));
        mainGradient.addColorStop(1, this.addAlphaToColor(this.player.color, 0.4));
        
        this.ctx.fillStyle = mainGradient;
        this.ctx.beginPath();
        this.ctx.arc(aimEndX, aimEndY, pointRadius * pointPulse, 0, Math.PI * 2);
        this.ctx.fill();
        
        // Crosshair at aim point
        this.ctx.strokeStyle = this.addAlphaToColor(this.player.color, 0.9);
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        const crossSize = 8 + chargeProgress * 8;
        this.ctx.moveTo(aimEndX - crossSize, aimEndY);
        this.ctx.lineTo(aimEndX + crossSize, aimEndY);
        this.ctx.moveTo(aimEndX, aimEndY - crossSize);
        this.ctx.lineTo(aimEndX, aimEndY + crossSize);
        this.ctx.stroke();
    }
    
    drawCursorIndicator() {
        // Use mouse position directly for cursor display
        const cursorX = this.mouseX;
        const cursorY = this.mouseY;
        
        // Only draw direction line when game is running (not paused) and NOT aiming
        if (this.isRunning && !this.isPaused && !this.player.isAiming) {
            const dx = cursorX - this.player.x;
            const dy = cursorY - this.player.y;
            const dist = Math.hypot(dx, dy);
            
            if (dist > 5) {
                // Draw direction line (faded)
                this.ctx.strokeStyle = this.addAlphaToColor(this.player.color, 0.3);
                this.ctx.lineWidth = 2;
                this.ctx.setLineDash([5, 5]);
                this.ctx.beginPath();
                this.ctx.moveTo(this.player.x, this.player.y);
                this.ctx.lineTo(cursorX, cursorY);
                this.ctx.stroke();
                this.ctx.setLineDash([]);
            }
        }
        
        // Draw cursor indicator (custom cursor design) - always visible
        const cursorSize = 16;
        const cursorRadius = 10;
        
        // Determine cursor color based on game state
        let cursorColor = this.player.color;
        let cursorAlpha = 0.8;
        
        if (this.isPaused) {
            // More visible during pause
            cursorAlpha = 0.9;
        } else if (!this.isRunning) {
            // Even more visible when not started
            cursorAlpha = 1.0;
        }
        
        // Outer glow
        const gradient = this.ctx.createRadialGradient(
            cursorX, cursorY, 0,
            cursorX, cursorY, cursorRadius * 2.5
        );
        gradient.addColorStop(0, this.addAlphaToColor(cursorColor, cursorAlpha * 0.5));
        gradient.addColorStop(1, this.addAlphaToColor(cursorColor, 0));
        
        this.ctx.fillStyle = gradient;
        this.ctx.beginPath();
        this.ctx.arc(cursorX, cursorY, cursorRadius * 2.5, 0, Math.PI * 2);
        this.ctx.fill();
        
        // Outer ring (thicker and more visible)
        this.ctx.strokeStyle = this.addAlphaToColor(cursorColor, cursorAlpha);
        this.ctx.lineWidth = 2.5;
        this.ctx.beginPath();
        this.ctx.arc(cursorX, cursorY, cursorRadius, 0, Math.PI * 2);
        this.ctx.stroke();
        
        // Inner crosshair (more visible)
        this.ctx.strokeStyle = this.addAlphaToColor(cursorColor, cursorAlpha);
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        // Horizontal line
        this.ctx.moveTo(cursorX - cursorSize / 2, cursorY);
        this.ctx.lineTo(cursorX + cursorSize / 2, cursorY);
        // Vertical line
        this.ctx.moveTo(cursorX, cursorY - cursorSize / 2);
        this.ctx.lineTo(cursorX, cursorY + cursorSize / 2);
        this.ctx.stroke();
        
        // Center dot (larger and more visible)
        this.ctx.fillStyle = this.addAlphaToColor(cursorColor, cursorAlpha);
        this.ctx.beginPath();
        this.ctx.arc(cursorX, cursorY, 3, 0, Math.PI * 2);
        this.ctx.fill();
    }
    
    draw() {
        // Clear canvas
        this.ctx.fillStyle = '#0a0a0a';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw grid pattern
        this.ctx.strokeStyle = '#111';
        this.ctx.lineWidth = 1;
        const gridSize = 50;
        
        for (let x = 0; x < this.canvas.width; x += gridSize) {
            this.ctx.beginPath();
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, this.canvas.height);
            this.ctx.stroke();
        }
        
        for (let y = 0; y < this.canvas.height; y += gridSize) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(this.canvas.width, y);
            this.ctx.stroke();
        }
        
        // Draw particles (with reduced opacity when paused)
        if (this.isPaused) {
            this.ctx.save();
            this.ctx.globalAlpha = 0.4;
            this.particles.forEach(particle => particle.draw(this.ctx));
            this.ctx.restore();
        } else {
            this.particles.forEach(particle => particle.draw(this.ctx));
        }
        
        // Calculate activation progress for hover feedback (only when not running and not paused)
        let activationProgress = 0;
        if (!this.isRunning && !this.isPaused && this.isHovering && this.hoverStartTime) {
            activationProgress = Math.min(1, (Date.now() - this.hoverStartTime) / this.hoverDuration);
        }
        
        // Draw hover radius indicator when not running and not paused
        if (!this.isRunning && !this.isPaused) {
            this.drawHoverRadius();
        }
        
        // Draw orbs (always visible)
        this.drawOrb(this.player.x, this.player.y, this.player.radius, this.player.color, this.player.trail, activationProgress);
        this.drawOrb(this.ai.x, this.ai.y, this.ai.radius, this.ai.color, this.ai.trail);
        
        // Draw dark overlay when paused (death screen)
        if (this.isPaused) {
            this.drawDeathOverlay();
        }
        
        // Draw custom cursor indicator - always visible (after overlay so it's on top)
        this.drawCursorIndicator();
        
        // Draw aiming indicator if aiming
        if (this.isRunning && !this.isPaused && this.player.isAiming) {
            this.drawAimingIndicator();
        }
        
        if (this.isRunning && !this.isPaused) {
            // Draw projectiles
            this.projectiles.forEach(projectile => projectile.draw(this.ctx));
            
            // Draw player projectiles
            this.playerProjectiles.forEach(projectile => projectile.draw(this.ctx));
            
            // Draw player beams
            this.playerBeams.forEach(beam => beam.draw(this.ctx));
            
            // Draw AI beams
            this.aiBeams.forEach(beam => beam.draw(this.ctx));
            
            // Draw AI prediction line
            const inputs = this.getNeuralNetworkInputs();
            const outputs = this.nn.predict(inputs);
            const moveX = (outputs[0] - 0.5) * 2;
            const moveY = (outputs[1] - 0.5) * 2;
            const predictX = this.ai.x + moveX * 50;
            const predictY = this.ai.y + moveY * 50;
            
            this.ctx.strokeStyle = '#ff4a4a20';
            this.ctx.lineWidth = 2;
            this.ctx.setLineDash([5, 5]);
            this.ctx.beginPath();
            this.ctx.moveTo(this.ai.x, this.ai.y);
            this.ctx.lineTo(predictX, predictY);
            this.ctx.stroke();
            this.ctx.setLineDash([]);
            
            // Draw dash windup indicator (charging phase) - CLEAR AND VISIBLE
            if (this.ai.isDashWindup) {
                const windupProgress = 1 - (this.ai.dashWindupTime / this.abilities.dash.windupDuration);
                const dashDistance = this.abilities.dash.dashDistance;
                
                // Use pre-calculated target position
                const dashTargetX = this.ai.dashTargetX;
                const dashTargetY = this.ai.dashTargetY;
                
                // 1. Draw full path line (very visible)
                this.ctx.strokeStyle = '#ffaa0060';
                this.ctx.lineWidth = 4;
                this.ctx.lineCap = 'round';
                this.ctx.setLineDash([]);
                this.ctx.beginPath();
                this.ctx.moveTo(this.ai.dashStartX, this.ai.dashStartY);
                this.ctx.lineTo(dashTargetX, dashTargetY);
                this.ctx.stroke();
                
                // 2. Draw progress fill line (shows charge progress)
                const progressLength = dashDistance * windupProgress;
                const progressEndX = this.ai.dashStartX + this.ai.dashDirection.x * progressLength;
                const progressEndY = this.ai.dashStartY + this.ai.dashDirection.y * progressLength;
                
                this.ctx.strokeStyle = '#ffaa00';
                this.ctx.lineWidth = 6;
                this.ctx.lineCap = 'round';
                this.ctx.beginPath();
                this.ctx.moveTo(this.ai.dashStartX, this.ai.dashStartY);
                this.ctx.lineTo(progressEndX, progressEndY);
                this.ctx.stroke();
                
                // 3. Draw arrowhead at target (always visible)
                const arrowSize = 20;
                const angle = Math.atan2(this.ai.dashDirection.y, this.ai.dashDirection.x);
                
                this.ctx.save();
                this.ctx.translate(dashTargetX, dashTargetY);
                this.ctx.rotate(angle);
                
                // Arrow body
                this.ctx.fillStyle = '#ffaa00';
                this.ctx.beginPath();
                this.ctx.moveTo(0, 0);
                this.ctx.lineTo(-arrowSize, -arrowSize * 0.6);
                this.ctx.lineTo(-arrowSize * 0.7, 0);
                this.ctx.lineTo(-arrowSize, arrowSize * 0.6);
                this.ctx.closePath();
                this.ctx.fill();
                
                // Arrow outline
                this.ctx.strokeStyle = '#ff8800';
                this.ctx.lineWidth = 2;
                this.ctx.stroke();
                this.ctx.restore();
                
                // 4. Draw destination circle (large and pulsing)
                const pulse = Math.sin(windupProgress * Math.PI * 6) * 0.2 + 0.8;
                const circleRadius = 15 + windupProgress * 10;
                
                // Outer glow
                const targetGradient = this.ctx.createRadialGradient(
                    dashTargetX, dashTargetY, 0,
                    dashTargetX, dashTargetY, circleRadius * 2
                );
                targetGradient.addColorStop(0, `rgba(255, 170, 0, ${0.8 * pulse})`);
                targetGradient.addColorStop(1, 'rgba(255, 170, 0, 0)');
                this.ctx.fillStyle = targetGradient;
                this.ctx.beginPath();
                this.ctx.arc(dashTargetX, dashTargetY, circleRadius * 2, 0, Math.PI * 2);
                this.ctx.fill();
                
                // Inner circle
                this.ctx.fillStyle = `rgba(255, 170, 0, ${0.6 * pulse})`;
                this.ctx.beginPath();
                this.ctx.arc(dashTargetX, dashTargetY, circleRadius, 0, Math.PI * 2);
                this.ctx.fill();
                
                // Circle outline
                this.ctx.strokeStyle = '#ffaa00';
                this.ctx.lineWidth = 3;
                this.ctx.beginPath();
                this.ctx.arc(dashTargetX, dashTargetY, circleRadius, 0, Math.PI * 2);
                this.ctx.stroke();
                
                // 5. Draw charge rings around AI (expanding)
                const chargeRadius = this.ai.radius + windupProgress * 25;
                const ringAlpha = 0.3 + windupProgress * 0.5;
                
                // Outer ring
                this.ctx.strokeStyle = `rgba(255, 170, 0, ${ringAlpha})`;
                this.ctx.lineWidth = 3;
                this.ctx.beginPath();
                this.ctx.arc(this.ai.x, this.ai.y, chargeRadius, 0, Math.PI * 2);
                this.ctx.stroke();
                
                // Inner pulsing ring
                const innerPulse = Math.sin(windupProgress * Math.PI * 8) * 0.3 + 0.7;
                this.ctx.strokeStyle = `rgba(255, 170, 0, ${innerPulse})`;
                this.ctx.lineWidth = 2;
                this.ctx.beginPath();
                this.ctx.arc(this.ai.x, this.ai.y, this.ai.radius + windupProgress * 15, 0, Math.PI * 2);
                this.ctx.stroke();
                
                // 6. Draw progress arc (like a loading circle)
                this.ctx.strokeStyle = '#ffaa00';
                this.ctx.lineWidth = 4;
                this.ctx.lineCap = 'round';
                this.ctx.beginPath();
                const startAngle = -Math.PI / 2;
                const endAngle = startAngle + windupProgress * Math.PI * 2;
                this.ctx.arc(this.ai.x, this.ai.y, this.ai.radius + 20, startAngle, endAngle);
                this.ctx.stroke();
            }
            
            // Draw beam windup indicator (for R ability)
            if (this.ai.isBeamWindup) {
                const windupProgress = 1 - (this.ai.beamWindupTime / this.abilities.slowProjectile.windupDuration);
                
                // Calculate beam end point (extend to map edge)
                const aimDistance = Math.max(this.canvas.width, this.canvas.height) * 2;
                const beamEndX = this.ai.x + this.ai.beamDirection.x * aimDistance;
                const beamEndY = this.ai.y + this.ai.beamDirection.y * aimDistance;
                
                // Draw preview beam line (faint)
                this.ctx.strokeStyle = '#ff99ff40';
                this.ctx.lineWidth = 2;
                this.ctx.lineCap = 'round';
                this.ctx.setLineDash([5, 5]);
                this.ctx.beginPath();
                this.ctx.moveTo(this.ai.x, this.ai.y);
                this.ctx.lineTo(beamEndX, beamEndY);
                this.ctx.stroke();
                this.ctx.setLineDash([]);
                
                // Draw charge rings around AI
                const ringRadius = this.ai.radius + windupProgress * 20;
                const ringAlpha = 0.2 + windupProgress * 0.4;
                
                this.ctx.strokeStyle = `rgba(255, 153, 255, ${ringAlpha})`;
                this.ctx.lineWidth = 2 + windupProgress * 2;
                this.ctx.beginPath();
                this.ctx.arc(this.ai.x, this.ai.y, ringRadius, 0, Math.PI * 2);
                this.ctx.stroke();
                
                // Draw progress arc
                this.ctx.strokeStyle = '#ff99ff';
                this.ctx.lineWidth = 4;
                this.ctx.lineCap = 'round';
                this.ctx.beginPath();
                const startAngle = -Math.PI / 2;
                const endAngle = startAngle + windupProgress * Math.PI * 2;
                this.ctx.arc(this.ai.x, this.ai.y, this.ai.radius + 20, startAngle, endAngle);
                this.ctx.stroke();
            }
            
            // Draw dash execution (during actual dash movement)
            if (this.ai.isDashing) {
                // Draw trail from start to current position
                this.ctx.strokeStyle = '#ffaa00';
                this.ctx.lineWidth = 8;
                this.ctx.lineCap = 'round';
                this.ctx.setLineDash([]);
                this.ctx.beginPath();
                this.ctx.moveTo(this.ai.dashStartX, this.ai.dashStartY);
                this.ctx.lineTo(this.ai.x, this.ai.y);
                this.ctx.stroke();
                
                // Draw remaining path (faint)
                this.ctx.strokeStyle = '#ffaa0040';
                this.ctx.lineWidth = 4;
                this.ctx.beginPath();
                this.ctx.moveTo(this.ai.x, this.ai.y);
                this.ctx.lineTo(this.ai.dashTargetX, this.ai.dashTargetY);
                this.ctx.stroke();
                
                // Draw current position glow
                const dashGlow = this.ctx.createRadialGradient(
                    this.ai.x, this.ai.y, 0,
                    this.ai.x, this.ai.y, this.ai.radius * 3
                );
                dashGlow.addColorStop(0, '#ffaa0080');
                dashGlow.addColorStop(1, '#ffaa0000');
                this.ctx.fillStyle = dashGlow;
                this.ctx.beginPath();
                this.ctx.arc(this.ai.x, this.ai.y, this.ai.radius * 3, 0, Math.PI * 2);
                this.ctx.fill();
            }
        }
    }
    
    animate(currentTime) {
        const deltaTime = (currentTime - this.lastTime) / 1000;
        this.lastTime = currentTime;
        
        this.update(deltaTime);
        this.draw();
        
        requestAnimationFrame((time) => this.animate(time));
    }
}

// Initialize game when page loads
let game;
window.addEventListener('load', () => {
    game = new Game();
});
