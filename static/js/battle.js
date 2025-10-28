/**
 * Warhammer: The Old World - Battle Interface JavaScript
 * Real-time canvas animations with WebSocket communication
 */

class WarhammerBattle {
    constructor() {
        this.socket = io();
        this.canvas = document.getElementById('battlefield');
        this.ctx = this.canvas.getContext('2d');
        this.battleState = null;
        this.scale = 1;
        this.animationQueue = [];
        this.selectedUnit = null;
        this.tooltip = document.getElementById('unit-tooltip');
        this.battleRunning = false;
        
        // Animation settings
        this.animations = {
            movement: [],
            projectiles: [],
            explosions: [],
            effects: []
        };
        
        this.setupEventListeners();
        this.setupCanvas();
        this.hideLoading();
        this.startAnimationLoop();
    }

    setupEventListeners() {
        // Socket events
        this.socket.on('connect', () => {
            this.updateConnectionStatus(true);
            this.logMessage('Connected to battle server', 'welcome');
        });

        this.socket.on('disconnect', () => {
            this.updateConnectionStatus(false);
            this.logMessage('Disconnected from server', 'combat');
        });

        this.socket.on('battle_update', (data) => {
            this.handleBattleUpdate(data);
        });

        // Live log stream
        this.socket.on('log_event', (data) => {
            if (data && data.message) {
                this.logMessage(data.message, 'combat');
            }
        });

        // Button events
        document.getElementById('new-battle-btn').addEventListener('click', () => {
            this.newBattle();
        });

        document.getElementById('start-battle-btn').addEventListener('click', () => {
            this.startBattle();
        });

        document.getElementById('pause-battle-btn').addEventListener('click', () => {
            this.pauseBattle();
        });

        document.getElementById('next-turn-btn').addEventListener('click', () => {
            this.nextTurn();
        });

        // Canvas events
        this.canvas.addEventListener('mousemove', (e) => {
            this.handleMouseMove(e);
        });

        this.canvas.addEventListener('click', (e) => {
            this.handleCanvasClick(e);
        });

        this.canvas.addEventListener('mouseleave', () => {
            this.hideTooltip();
        });

        // Window resize
        window.addEventListener('resize', () => {
            this.setupCanvas();
        });
    }

    setupCanvas() {
        const container = this.canvas.parentElement;
        const rect = container.getBoundingClientRect();
        
        // Set canvas size to container size
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;
        
        // Calculate scale to fit battlefield with padding (dynamic width/height)
        const padding = 50;
        const bw = (this.battleState && this.battleState.battlefield && this.battleState.battlefield.width) ? this.battleState.battlefield.width : 68;
        const bh = (this.battleState && this.battleState.battlefield && this.battleState.battlefield.height) ? this.battleState.battlefield.height : 48;
        this.scaleX = (this.canvas.width - padding * 2) / bw;
        this.scaleY = (this.canvas.height - padding * 2) / bh;
        this.scale = Math.min(this.scaleX, this.scaleY);
        
        // Calculate offset to center the battlefield
        this.offsetX = (this.canvas.width - bw * this.scale) / 2;
        this.offsetY = (this.canvas.height - bh * this.scale) / 2;
        
        console.log(`Canvas: ${this.canvas.width}x${this.canvas.height}, Scale: ${this.scale}, Offset: ${this.offsetX}, ${this.offsetY}`);
        
        this.redrawBattlefield();
    }

    updateConnectionStatus(connected) {
        const indicator = document.getElementById('connection-indicator');
        const text = document.getElementById('connection-text');
        
        if (connected) {
            indicator.textContent = '🟢';
            text.textContent = 'Connected';
        } else {
            indicator.textContent = '🔴';
            text.textContent = 'Disconnected';
        }
    }

    hideLoading() {
        const overlay = document.getElementById('loading-overlay');
        setTimeout(() => {
            overlay.style.opacity = '0';
            setTimeout(() => {
                overlay.style.display = 'none';
            }, 500);
        }, 1000);
    }

    async newBattle() {
        this.showLoading('Setting up new battle...');
        console.log('Creating new battle...');
        
        try {
            const response = await fetch('/api/new_battle', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const data = await response.json();
            console.log('Battle created:', data);
            
            if (data.success) {
                this.battleState = data.battle_state;
                console.log('Battle state:', this.battleState);
                this.updateUI();
                this.redrawBattlefield();
                this.logMessage('New battle prepared! Armies deployed.', 'welcome');
            }
        } catch (error) {
            console.error('Failed to create new battle:', error);
            this.logMessage('Failed to create battle', 'combat');
        }
        
        this.hideLoading();
    }

    async startBattle() {
        if (!this.battleState) {
            this.logMessage('Create a new battle first!', 'phase');
            return;
        }
        
        this.showLoading('Starting battle...');
        console.log('Starting battle...');
        
        try {
            const response = await fetch('/api/start_battle', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const data = await response.json();
            console.log('Battle started:', data);
            
            if (data && data.success) {
                this.logMessage('Battle commenced! May the dice favor you.', 'movement');
                // Start polling for battle updates
                this.pollBattleUpdates();
            } else {
                this.logMessage('Failed to start battle (server returned false)', 'combat');
            }
        } catch (error) {
            console.error('Failed to start battle:', error);
            this.logMessage('Failed to start battle', 'combat');
        }
        
        this.hideLoading();
    }

    pauseBattle() {
        this.battleRunning = false;
        this.logMessage('Battle paused', 'phase');
    }

    nextTurn() {
        // This will be handled by polling
        this.logMessage('Next turn...', 'phase');
    }

    async pollBattleUpdates() {
        this.battleRunning = true;
        
        while (this.battleRunning) {
            try {
                const response = await fetch('/api/battle_state');
                const data = await response.json();
                
                if (data) {
                    const oldState = this.battleState;
                    this.battleState = data;
                    this.updateUI();
                    this.redrawBattlefield();
                    
                    // Check if battle ended
                    if (data.battle_state === 'completed') {
                        this.battleRunning = false;
                        this.handleBattleEnd();
                        break;
                    }
                }
                
                // Wait 1 second before next poll
                await new Promise(resolve => setTimeout(resolve, 1000));
                
            } catch (error) {
                console.error('Failed to get battle state:', error);
                await new Promise(resolve => setTimeout(resolve, 2000));
            }
        }
    }

    handleBattleUpdate(data) {
        this.battleState = data.battle_state;
        this.updateUI();
        
        switch (data.event_type) {
            case 'phase_change':
                this.handlePhaseChange(data.data.phase);
                break;
            case 'movement_complete':
                this.animateMovement();
                break;
            case 'shooting_complete':
                this.animateShooting();
                break;
            case 'charge_complete':
                this.animateCharges();
                break;
            case 'combat_complete':
                this.animateCombat();
                break;
            case 'battle_end':
                this.handleBattleEnd();
                break;
        }
        
        this.redrawBattlefield();
    }

    handlePhaseChange(phase) {
        this.logMessage(`${phase} Phase begins`, 'phase');
        this.updateActivePhase(phase);
    }

    updateActivePhase(phase) {
        document.querySelectorAll('.phase-item').forEach(item => {
            item.classList.remove('active');
            if (item.dataset.phase === phase) {
                item.classList.add('active');
            }
        });
    }

    updateUI() {
        if (!this.battleState) return;
        
        // Update turn info
        document.getElementById('current-turn').textContent = this.battleState.turn;
        document.getElementById('current-phase').textContent = this.battleState.phase;
        const playerName = this.battleState.current_player === 1 ? 'Empire' : 'Orcs';
        document.getElementById('current-player').textContent = playerName;
        
        // Update army lists
        this.updateArmyList('empire-units', this.battleState.units.filter(u => u.player === 1));
        this.updateArmyList('orc-units', this.battleState.units.filter(u => u.player === 2));
        
        // Update battle log
        if (this.battleState.battle_events) {
            // Only show new log entries
            const logContainer = document.getElementById('battle-log');
            const currentEntries = logContainer.children.length;
            for (let i = currentEntries; i < this.battleState.battle_events.length; i++) {
                this.logMessage(this.battleState.battle_events[i], 'combat');
            }
        }
    }

    updateArmyList(elementId, units) {
        const container = document.getElementById(elementId);
        container.innerHTML = '';
        
        units.forEach((unit) => {
            const unitCard = document.createElement('div');
            unitCard.className = `unit-card ${unit.is_alive ? '' : 'unit-critical'}`;
            unitCard.dataset.unitId = unit.id;
            
            const healthClass = unit.models / unit.max_models > 0.7 ? 'unit-health' :
                              unit.models / unit.max_models > 0.3 ? 'unit-damaged' : 'unit-critical';
            
            unitCard.innerHTML = `
                <div class="unit-name">${unit.name}</div>
                <div class="unit-stats">
                    <span class="${healthClass}">${unit.models}/${unit.max_models} models</span>
                    <span>${unit.unit_type} | ${unit.formation}</span>
                    <span>${unit.points} pts</span>
                </div>
            `;
            
            unitCard.addEventListener('click', () => {
                this.selectUnit(unit);
            });
            
            container.appendChild(unitCard);
        });
    }

    selectUnit(unit) {
        this.selectedUnit = unit;
        
        // Highlight selected unit card
        document.querySelectorAll('.unit-card').forEach(card => {
            card.classList.remove('selected');
        });
        
        // Highlight by unique unit id
        const unitCard = document.querySelector(`[data-unit-id="${unit.id}"]`);
        if (unitCard) {
            unitCard.classList.add('selected');
        }
        
        this.redrawBattlefield();
    }

    redrawBattlefield() {
        if (!this.battleState) return;
        
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw terrain
        this.drawTerrain();
        
        // Draw grid
        this.drawGrid();
        
        // Draw units
        this.battleState.units.forEach(unit => {
            if (unit.is_alive) {
                this.drawUnit(unit);
                if (unit.has_charged || unit.is_engaged) {
                    this.highlightEngaged(unit);
                }
            }
        });
        
        // Draw animations
        this.drawAnimations();
    }

    drawTerrain() {
        if (!this.battleState.terrain) return;
        
        this.battleState.terrain.forEach(feature => {
            this.ctx.save();
            
            const x = this.offsetX + feature.x * this.scale;
            const y = this.offsetY + feature.y * this.scale;
            const width = feature.width * this.scale;
            const height = feature.height * this.scale;
            
            this.ctx.fillStyle = feature.color;
            this.ctx.globalAlpha = 0.7;
            
            if (feature.type === 'hill') {
                // Draw ellipse for hill
                this.ctx.beginPath();
                this.ctx.ellipse(x + width/2, y + height/2, width/2, height/2, 0, 0, 2 * Math.PI);
                this.ctx.fill();
            } else {
                // Draw rectangle for other terrain
                this.ctx.fillRect(x, y, width, height);
            }
            
            // Add terrain label
            this.ctx.globalAlpha = 1;
            this.ctx.fillStyle = '#fff';
            this.ctx.font = `bold ${Math.max(10, this.scale * 0.8)}px Cinzel`;
            this.ctx.textAlign = 'center';
            this.ctx.shadowColor = 'black';
            this.ctx.shadowBlur = 3;
            this.ctx.fillText(feature.name, x + width/2, y + height/2);
            this.ctx.shadowBlur = 0;
            
            this.ctx.restore();
        });
    }

    drawGrid() {
        this.ctx.save();
        this.ctx.strokeStyle = 'rgba(74, 109, 42, 0.4)';
        this.ctx.lineWidth = 1;
        
        const bw = (this.battleState && this.battleState.battlefield && this.battleState.battlefield.width) ? this.battleState.battlefield.width : 68;
        const bh = (this.battleState && this.battleState.battlefield && this.battleState.battlefield.height) ? this.battleState.battlefield.height : 48;
        // Vertical lines (every 6 inches)
        for (let i = 0; i <= bw; i += 6) {
            const x = this.offsetX + i * this.scale;
            this.ctx.beginPath();
            this.ctx.moveTo(x, this.offsetY);
            this.ctx.lineTo(x, this.offsetY + bh * this.scale);
            this.ctx.stroke();
            
            // Add distance markers
            if (i > 0) {
                this.ctx.fillStyle = 'rgba(144, 238, 144, 0.7)';
                this.ctx.font = `${Math.max(8, this.scale * 0.4)}px Arial`;
                this.ctx.textAlign = 'center';
                this.ctx.fillText(`${i}"`, x, this.offsetY - 5);
            }
        }
        
        // Horizontal lines (every 6 inches)
        for (let i = 0; i <= bh; i += 6) {
            const y = this.offsetY + i * this.scale;
            this.ctx.beginPath();
            this.ctx.moveTo(this.offsetX, y);
            this.ctx.lineTo(this.offsetX + bw * this.scale, y);
            this.ctx.stroke();
            
            // Add distance markers
            if (i > 0) {
                this.ctx.fillStyle = 'rgba(144, 238, 144, 0.7)';
                this.ctx.font = `${Math.max(8, this.scale * 0.4)}px Arial`;
                this.ctx.textAlign = 'left';
                this.ctx.fillText(`${i}"`, this.offsetX - 20, y + 3);
            }
        }
        
        this.ctx.restore();
    }

    drawUnit(unit) {
        this.ctx.save();
        
        // Convert position (inches) to screen coordinates
        const x = this.offsetX + unit.x * this.scale;
        const y = this.offsetY + unit.y * this.scale;
        
        // Draw formation rectangle
        const formationWidth = unit.width * this.scale * 0.8;  // Model spacing
        const formationHeight = unit.depth * this.scale * 0.8;
        
        // Formation background - determine color based on faction
        this.ctx.fillStyle = unit.color ? unit.color : (unit.player === 1 ? 'rgba(79, 195, 247, 0.8)' : 'rgba(129, 199, 132, 0.8)');
        this.ctx.globalAlpha = 0.6;
        this.ctx.fillRect(x - formationWidth/2, y - formationHeight/2, formationWidth, formationHeight);
        
        // Formation border
        this.ctx.globalAlpha = 1;
        this.ctx.strokeStyle = this.selectedUnit && this.selectedUnit.id === unit.id ? '#f7d060' : '#fff';
        this.ctx.lineWidth = this.selectedUnit && this.selectedUnit.id === unit.id ? 3 : 2;
        this.ctx.strokeRect(x - formationWidth/2, y - formationHeight/2, formationWidth, formationHeight);
        
        // Draw individual models
        this.drawModels(unit, x, y);
        
        // Draw unit name
        this.ctx.fillStyle = '#fff';
        this.ctx.font = `bold ${Math.max(10, this.scale * 0.5)}px Cinzel`;
        this.ctx.textAlign = 'center';
        this.ctx.shadowColor = 'black';
        this.ctx.shadowBlur = 2;
        this.ctx.fillText(unit.name, x, y - formationHeight/2 - 15);
        this.ctx.shadowBlur = 0;
        
        // Draw health bar
        this.drawHealthBar(unit, x, y + formationHeight/2 + 10);
        
        this.ctx.restore();
    }

    drawModels(unit, centerX, centerY) {
        const modelSize = Math.max(2, this.scale * 0.15);
        const spacing = Math.max(4, this.scale * 0.2);
        
        let modelsDrawn = 0;
        for (let rank = 0; rank < unit.depth && modelsDrawn < unit.models; rank++) {
            for (let file = 0; file < unit.width && modelsDrawn < unit.models; file++) {
                const modelX = centerX - (unit.width * spacing) / 2 + file * spacing + spacing/2;
                const modelY = centerY - (unit.depth * spacing) / 2 + rank * spacing + spacing/2;
                
                this.ctx.fillStyle = unit.faction === "nuln" ? '#4fc3f7' : '#81c784';
                this.ctx.beginPath();
                this.ctx.arc(modelX, modelY, modelSize, 0, 2 * Math.PI);
                this.ctx.fill();
                
                this.ctx.strokeStyle = '#fff';
                this.ctx.lineWidth = 1;
                this.ctx.stroke();
                
                modelsDrawn++;
            }
        }
    }

    drawHealthBar(unit, x, y) {
        const barWidth = Math.max(40, this.scale * 1.5);
        const barHeight = 6;
        const healthPercent = unit.models / unit.max_models;
        
        // Background
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
        this.ctx.fillRect(x - barWidth/2, y, barWidth, barHeight);
        
        // Health
        const healthColor = healthPercent > 0.7 ? '#28a745' : 
                           healthPercent > 0.3 ? '#ffc107' : '#dc3545';
        this.ctx.fillStyle = healthColor;
        this.ctx.fillRect(x - barWidth/2, y, barWidth * healthPercent, barHeight);
        
        // Border
        this.ctx.strokeStyle = '#fff';
        this.ctx.lineWidth = 1;
        this.ctx.strokeRect(x - barWidth/2, y, barWidth, barHeight);
        
        // Health text
        this.ctx.fillStyle = '#fff';
        this.ctx.font = `${Math.max(8, this.scale * 0.3)}px Arial`;
        this.ctx.textAlign = 'center';
        this.ctx.fillText(`${unit.models}/${unit.max_models}`, x, y + barHeight + 12);
    }

    highlightEngaged(unit) {
        const x = this.offsetX + unit.x * this.scale;
        const y = this.offsetY + unit.y * this.scale;
        const radius = Math.max(unit.width, unit.depth) * this.scale * 0.6;
        this.ctx.save();
        this.ctx.strokeStyle = 'rgba(255, 215, 0, 0.8)';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        this.ctx.arc(x, y, radius, 0, 2 * Math.PI);
        this.ctx.stroke();
        this.ctx.restore();
    }

    drawAnimations() {
        // Draw projectiles
        this.animations.projectiles.forEach(projectile => {
            this.ctx.save();
            this.ctx.strokeStyle = '#ff6600';
            this.ctx.lineWidth = 3;
            this.ctx.globalAlpha = projectile.alpha;
            
            const startX = this.offsetX + projectile.startX * this.scale;
            const startY = this.offsetY + projectile.startY * this.scale;
            const endX = this.offsetX + projectile.x * this.scale;
            const endY = this.offsetY + projectile.y * this.scale;
            
            this.ctx.beginPath();
            this.ctx.moveTo(startX, startY);
            this.ctx.lineTo(endX, endY);
            this.ctx.stroke();
            
            this.ctx.restore();
        });
        
        // Draw explosions
        this.animations.explosions.forEach(explosion => {
            this.ctx.save();
            this.ctx.fillStyle = `rgba(255, 100, 0, ${explosion.alpha})`;
            
            const x = this.offsetX + explosion.x * this.scale;
            const y = this.offsetY + explosion.y * this.scale;
            const radius = explosion.radius * this.scale;
            
            this.ctx.beginPath();
            this.ctx.arc(x, y, radius, 0, 2 * Math.PI);
            this.ctx.fill();
            
            this.ctx.restore();
        });
    }

    animateMovement() {
        this.logMessage('Units repositioning...', 'movement');
        // Movement animation handled by redrawing
    }

    animateShooting() {
        if (!this.battleState.units) return;
        
        // Find shooting units and create projectile animations
        const shooters = this.battleState.units.filter(u => u.weapon_range > 0 && u.player === this.battleState.current_player);
        
        shooters.forEach(shooter => {
            const targets = this.battleState.units.filter(u => u.player !== shooter.player && u.is_alive);
            if (targets.length > 0) {
                const target = targets[Math.floor(Math.random() * targets.length)];
                
                this.createProjectile(shooter.x, shooter.y, target.x, target.y);
                this.createExplosion(target.x, target.y);
            }
        });
        
        this.logMessage('Volleys exchanged!', 'combat');
    }

    animateCharges() {
        this.logMessage('Cavalry thunders across the field!', 'movement');
        // Charge animation handled by redrawing with updated positions
    }

    animateCombat() {
        if (!this.battleState.units) return;
        
        // Create combat effects for engaged units
        this.battleState.units.forEach(unit => {
            if (unit.is_alive) {
                const enemies = this.battleState.units.filter(u => 
                    u.player !== unit.player && u.is_alive && 
                    Math.sqrt((u.x - unit.x)**2 + (u.y - unit.y)**2) <= 3
                );
                
                if (enemies.length > 0) {
                    this.createCombatEffect(unit.x, unit.y);
                }
            }
        });
        
        this.logMessage('Melee combat rages!', 'combat');
    }

    createProjectile(startX, startY, endX, endY) {
        const projectile = {
            startX, startY, x: startX, y: startY,
            endX, endY, alpha: 1, speed: 0.1
        };
        
        this.animations.projectiles.push(projectile);
        
        const animate = () => {
            projectile.x += (projectile.endX - projectile.startX) * projectile.speed;
            projectile.y += (projectile.endY - projectile.startY) * projectile.speed;
            projectile.alpha -= 0.02;
            
            if (projectile.alpha <= 0 || 
                Math.abs(projectile.x - projectile.endX) < 1) {
                const index = this.animations.projectiles.indexOf(projectile);
                if (index > -1) this.animations.projectiles.splice(index, 1);
            } else {
                requestAnimationFrame(animate);
            }
        };
        
        animate();
    }

    createExplosion(x, y) {
        const explosion = {
            x, y, radius: 0.5, alpha: 1
        };
        
        this.animations.explosions.push(explosion);
        
        const animate = () => {
            explosion.radius += 0.2;
            explosion.alpha -= 0.05;
            
            if (explosion.alpha <= 0) {
                const index = this.animations.explosions.indexOf(explosion);
                if (index > -1) this.animations.explosions.splice(index, 1);
            } else {
                requestAnimationFrame(animate);
            }
        };
        
        animate();
    }

    createCombatEffect(x, y) {
        // Simple sparkle effect for combat
        for (let i = 0; i < 5; i++) {
            setTimeout(() => {
                this.createExplosion(
                    x + (Math.random() - 0.5) * 2,
                    y + (Math.random() - 0.5) * 2
                );
            }, i * 100);
        }
    }

    handleMouseMove(e) {
        if (!this.battleState) return;
        
        const rect = this.canvas.getBoundingClientRect();
        const mouseX = (e.clientX - rect.left - this.offsetX) / this.scale;
        const mouseY = (e.clientY - rect.top - this.offsetY) / this.scale;
        
        // Find unit under mouse
        const hoveredUnit = this.battleState.units.find(unit => {
            if (!unit.is_alive) return false;
            
            const distance = Math.sqrt((unit.x - mouseX)**2 + (unit.y - mouseY)**2);
            return distance <= Math.max(unit.width, unit.depth) * 2; // Increased hit area
        });
        
        if (hoveredUnit) {
            this.showTooltip(hoveredUnit, e.clientX, e.clientY);
        } else {
            this.hideTooltip();
        }
    }

    handleCanvasClick(e) {
        const rect = this.canvas.getBoundingClientRect();
        const mouseX = (e.clientX - rect.left - this.offsetX) / this.scale;
        const mouseY = (e.clientY - rect.top - this.offsetY) / this.scale;
        
        console.log(`Click at: ${mouseX.toFixed(1)}, ${mouseY.toFixed(1)}`);
        
        if (this.battleState) {
            const clickedUnit = this.battleState.units.find(unit => {
                if (!unit.is_alive) return false;
                
                const distance = Math.sqrt((unit.x - mouseX)**2 + (unit.y - mouseY)**2);
                return distance <= Math.max(unit.width, unit.depth) * 2; // Increased hit area
            });
            
            if (clickedUnit) {
                console.log(`Clicked unit: ${clickedUnit.name}`);
                this.selectUnit(clickedUnit);
            }
        }
    }

    showTooltip(unit, x, y) {
        const tooltip = document.getElementById('unit-tooltip');
        
        // Update basic unit information
        document.getElementById('tooltip-name').textContent = unit.name;
        document.getElementById('tooltip-models').textContent = unit.models;
        document.getElementById('tooltip-max').textContent = unit.max_models;
        document.getElementById('tooltip-formation').textContent = 
            `${unit.formation} (${unit.width}x${unit.depth})`;
        
        // Update statistics
        document.getElementById('tooltip-movement').textContent = unit.movement;
        document.getElementById('tooltip-ws').textContent = unit.weapon_skill;
        document.getElementById('tooltip-bs').textContent = unit.ballistic_skill;
        document.getElementById('tooltip-strength').textContent = unit.strength;
        document.getElementById('tooltip-toughness').textContent = unit.toughness;
        document.getElementById('tooltip-wounds').textContent = unit.wounds;
        document.getElementById('tooltip-attacks').textContent = unit.attacks;
        document.getElementById('tooltip-leadership').textContent = unit.leadership;
        document.getElementById('tooltip-armor').textContent = unit.armor_save + '+';
        
        // Handle range display
        const rangeElement = document.getElementById('tooltip-range');
        if (unit.weapon_range && unit.weapon_range > 0) {
            document.getElementById('tooltip-range-value').textContent = unit.weapon_range;
            rangeElement.style.display = 'block';
        } else {
            rangeElement.style.display = 'none';
        }
        
        // Handle special rules
        const specialRules = [
            { id: 'special-standard', property: 'has_standard' },
            { id: 'special-musician', property: 'has_musician' },
            { id: 'special-armor-piercing', property: 'armor_piercing' },
            { id: 'special-frenzy', property: 'frenzy' },
            { id: 'special-fast-cavalry', property: 'fast_cavalry' },
            { id: 'special-lance', property: 'lance_formation' },
            { id: 'special-fear', property: 'fear' },
            { id: 'special-terror', property: 'terror' },
            { id: 'special-immune-fear', property: 'immune_to_fear' },
            { id: 'special-stubborn', property: 'stubborn' },
            { id: 'special-regeneration', property: 'regeneration' },
            { id: 'special-fleeing', property: 'is_fleeing' }
        ];
        
        specialRules.forEach(rule => {
            const element = document.getElementById(rule.id);
            if (unit[rule.property]) {
                element.style.display = 'block';
            } else {
                element.style.display = 'none';
            }
        });
        
        // Position and show tooltip
        tooltip.style.left = (x + 20) + 'px';
        tooltip.style.top = y + 'px';
        tooltip.style.display = 'block';
    }

    hideTooltip() {
        this.tooltip.style.display = 'none';
    }

    logMessage(message, type = 'normal') {
        const log = document.getElementById('battle-log');
        const entry = document.createElement('div');
        entry.className = `log-entry ${type}`;
        entry.textContent = message;
        
        log.appendChild(entry);
        log.scrollTop = log.scrollHeight;
        
        // Keep only last 50 entries
        while (log.children.length > 50) {
            log.removeChild(log.firstChild);
        }
    }

    showLoading(message) {
        const overlay = document.getElementById('loading-overlay');
        const text = overlay.querySelector('.loading-text');
        text.textContent = message;
        overlay.style.display = 'flex';
        overlay.style.opacity = '1';
    }

    startAnimationLoop() {
        const animate = () => {
            this.redrawBattlefield();
            requestAnimationFrame(animate);
        };
        animate();
    }

    handleBattleEnd() {
        this.logMessage('⚔️ BATTLE CONCLUDED! ⚔️', 'welcome');
        
        // Show victory message
        setTimeout(() => {
            const p1Alive = this.battleState.units.some(u => u.player === 1 && u.is_alive);
            const p2Alive = this.battleState.units.some(u => u.player === 2 && u.is_alive);
            
            if (!p1Alive) {
                this.logMessage('🏆 GLORIOUS VICTORY FOR THE ORCS!', 'combat');
            } else if (!p2Alive) {
                this.logMessage('🏆 THE EMPIRE STANDS TRIUMPHANT!', 'movement');
            } else {
                this.logMessage('🤝 A HARD-FOUGHT STALEMATE!', 'phase');
            }
        }, 1000);
    }
}

// Initialize the battle when page loads
document.addEventListener('DOMContentLoaded', () => {
    const battle = new WarhammerBattle();
    window.warhammerBattle = battle; // For debugging
}); 