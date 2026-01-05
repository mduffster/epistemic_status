/**
 * Matt Duffy Personal Website
 * Main JavaScript - GSAP Animations & Interactive Elements
 */

// =============================================================================
// Background Canvas - Marble to Digital Particle System
// =============================================================================

class ParticleBackground {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.particles = [];
        this.connections = [];
        this.mouse = { x: null, y: null, radius: 150 };
        this.animationId = null;

        // Colors from our palette
        this.colors = {
            gold: 'rgba(212, 168, 83, ',
            teal: 'rgba(45, 139, 164, ',
            marble: 'rgba(224, 220, 212, ',
            navy: 'rgba(30, 58, 95, '
        };

        this.init();
        this.bindEvents();
        this.animate();
    }

    init() {
        this.resize();
        this.createParticles();
    }

    resize() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
    }

    createParticles() {
        this.particles = [];
        const numberOfParticles = Math.floor((this.canvas.width * this.canvas.height) / 15000);

        for (let i = 0; i < numberOfParticles; i++) {
            const x = Math.random() * this.canvas.width;
            const y = Math.random() * this.canvas.height;
            const size = Math.random() * 2 + 0.5;
            const speedX = (Math.random() - 0.5) * 0.5;
            const speedY = (Math.random() - 0.5) * 0.5;

            // Assign colors based on position (marble effect at top, digital at bottom)
            const yRatio = y / this.canvas.height;
            let color;
            if (yRatio < 0.3) {
                color = Math.random() > 0.5 ? this.colors.marble : this.colors.gold;
            } else if (yRatio > 0.7) {
                color = Math.random() > 0.5 ? this.colors.teal : this.colors.navy;
            } else {
                // Transition zone
                const colors = [this.colors.gold, this.colors.teal, this.colors.marble];
                color = colors[Math.floor(Math.random() * colors.length)];
            }

            this.particles.push({
                x, y, size, speedX, speedY, color,
                baseX: x,
                baseY: y,
                density: Math.random() * 30 + 1
            });
        }
    }

    bindEvents() {
        window.addEventListener('resize', () => {
            this.resize();
            this.createParticles();
        });

        window.addEventListener('mousemove', (e) => {
            this.mouse.x = e.x;
            this.mouse.y = e.y;
        });

        window.addEventListener('mouseout', () => {
            this.mouse.x = null;
            this.mouse.y = null;
        });
    }

    animate() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Update and draw particles
        for (let i = 0; i < this.particles.length; i++) {
            const p = this.particles[i];

            // Mouse interaction
            if (this.mouse.x !== null && this.mouse.y !== null) {
                const dx = this.mouse.x - p.x;
                const dy = this.mouse.y - p.y;
                const distance = Math.sqrt(dx * dx + dy * dy);

                if (distance < this.mouse.radius) {
                    const force = (this.mouse.radius - distance) / this.mouse.radius;
                    const directionX = dx / distance;
                    const directionY = dy / distance;
                    p.x -= directionX * force * p.density * 0.5;
                    p.y -= directionY * force * p.density * 0.5;
                }
            }

            // Return to base position
            const dx = p.baseX - p.x;
            const dy = p.baseY - p.y;
            p.x += dx * 0.02;
            p.y += dy * 0.02;

            // Base movement
            p.baseX += p.speedX;
            p.baseY += p.speedY;

            // Bounce off edges
            if (p.baseX < 0 || p.baseX > this.canvas.width) p.speedX *= -1;
            if (p.baseY < 0 || p.baseY > this.canvas.height) p.speedY *= -1;

            // Keep in bounds
            p.baseX = Math.max(0, Math.min(this.canvas.width, p.baseX));
            p.baseY = Math.max(0, Math.min(this.canvas.height, p.baseY));

            // Draw particle
            this.ctx.beginPath();
            this.ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            this.ctx.fillStyle = p.color + '0.6)';
            this.ctx.fill();
        }

        // Draw connections
        this.drawConnections();

        this.animationId = requestAnimationFrame(() => this.animate());
    }

    drawConnections() {
        for (let i = 0; i < this.particles.length; i++) {
            for (let j = i + 1; j < this.particles.length; j++) {
                const dx = this.particles[i].x - this.particles[j].x;
                const dy = this.particles[i].y - this.particles[j].y;
                const distance = Math.sqrt(dx * dx + dy * dy);

                if (distance < 120) {
                    const opacity = (120 - distance) / 120 * 0.15;
                    this.ctx.beginPath();
                    this.ctx.strokeStyle = this.colors.gold + opacity + ')';
                    this.ctx.lineWidth = 0.5;
                    this.ctx.moveTo(this.particles[i].x, this.particles[i].y);
                    this.ctx.lineTo(this.particles[j].x, this.particles[j].y);
                    this.ctx.stroke();
                }
            }
        }
    }
}


// =============================================================================
// GSAP Animations
// =============================================================================

function initAnimations() {
    gsap.registerPlugin(ScrollTrigger);

    // Hero animations
    const heroTimeline = gsap.timeline({ defaults: { ease: 'power3.out' } });

    heroTimeline
        .to('.title-line', {
            opacity: 1,
            y: 0,
            duration: 1,
            stagger: 0.2
        })
        .to('.hero-tagline', {
            opacity: 1,
            duration: 0.8
        }, '-=0.4')
        .to('.hero-subtitle', {
            opacity: 1,
            duration: 0.8
        }, '-=0.4')
        .to('.hero-scroll', {
            opacity: 1,
            duration: 0.8
        }, '-=0.2');

    // Navigation scroll effect
    ScrollTrigger.create({
        start: 'top -100',
        onUpdate: (self) => {
            const nav = document.getElementById('nav');
            if (self.direction === 1 && self.progress > 0) {
                nav.classList.add('scrolled');
            } else if (self.progress === 0) {
                nav.classList.remove('scrolled');
            }
        }
    });

    // Section header animations
    gsap.utils.toArray('.section-header').forEach(header => {
        gsap.from(header, {
            scrollTrigger: {
                trigger: header,
                start: 'top 80%',
                toggleActions: 'play none none reverse'
            },
            opacity: 0,
            y: 50,
            duration: 0.8,
            ease: 'power3.out'
        });
    });

    // About section
    gsap.from('.about-lead', {
        scrollTrigger: {
            trigger: '.about-lead',
            start: 'top 80%',
            toggleActions: 'play none none reverse'
        },
        opacity: 0,
        y: 30,
        duration: 0.8
    });

    gsap.from('.about-text p', {
        scrollTrigger: {
            trigger: '.about-text',
            start: 'top 75%',
            toggleActions: 'play none none reverse'
        },
        opacity: 0,
        y: 30,
        duration: 0.6,
        stagger: 0.2
    });

    gsap.from('.detail-card', {
        scrollTrigger: {
            trigger: '.about-details',
            start: 'top 80%',
            toggleActions: 'play none none reverse'
        },
        opacity: 0,
        x: 30,
        duration: 0.6,
        stagger: 0.15
    });

    // Research cards
    gsap.from('.research-card', {
        scrollTrigger: {
            trigger: '.research-grid',
            start: 'top 80%',
            toggleActions: 'play none none reverse'
        },
        opacity: 0,
        y: 50,
        duration: 0.8,
        stagger: 0.2
    });

    // Writing section
    gsap.from('.substack-card', {
        scrollTrigger: {
            trigger: '.substack-card',
            start: 'top 80%',
            toggleActions: 'play none none reverse'
        },
        opacity: 0,
        x: -50,
        duration: 0.8
    });

    gsap.from('.lesswrong-card', {
        scrollTrigger: {
            trigger: '.lesswrong-card',
            start: 'top 80%',
            toggleActions: 'play none none reverse'
        },
        opacity: 0,
        x: 50,
        duration: 0.8
    });

    // CV items
    gsap.from('.cv-item', {
        scrollTrigger: {
            trigger: '.cv-content',
            start: 'top 80%',
            toggleActions: 'play none none reverse'
        },
        opacity: 0,
        y: 30,
        duration: 0.6,
        stagger: 0.15
    });

    gsap.from('.skill-category', {
        scrollTrigger: {
            trigger: '.skills-grid',
            start: 'top 85%',
            toggleActions: 'play none none reverse'
        },
        opacity: 0,
        y: 20,
        duration: 0.5,
        stagger: 0.1
    });

    // Contact links
    gsap.from('.contact-link', {
        scrollTrigger: {
            trigger: '.contact-links',
            start: 'top 85%',
            toggleActions: 'play none none reverse'
        },
        opacity: 0,
        y: 20,
        duration: 0.5,
        stagger: 0.1
    });

    // Parallax effects
    gsap.to('.hero-marble', {
        scrollTrigger: {
            trigger: '.hero',
            start: 'top top',
            end: 'bottom top',
            scrub: 1
        },
        y: 200,
        opacity: 0
    });

    // Card marble accents subtle animation
    gsap.utils.toArray('.card-marble').forEach(marble => {
        gsap.to(marble, {
            scrollTrigger: {
                trigger: marble,
                start: 'top bottom',
                end: 'bottom top',
                scrub: 1
            },
            backgroundPosition: '200% center'
        });
    });
}


// =============================================================================
// Navigation
// =============================================================================

function initNavigation() {
    const navToggle = document.querySelector('.nav-toggle');
    const navLinks = document.querySelector('.nav-links');

    if (navToggle) {
        navToggle.addEventListener('click', () => {
            navLinks.classList.toggle('active');
            navToggle.classList.toggle('active');
        });
    }

    // Close mobile nav on link click
    document.querySelectorAll('.nav-links a').forEach(link => {
        link.addEventListener('click', () => {
            navLinks.classList.remove('active');
            navToggle.classList.remove('active');
        });
    });

    // Smooth scroll for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                const navHeight = document.querySelector('.nav').offsetHeight;
                const targetPosition = target.offsetTop - navHeight;
                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth'
                });
            }
        });
    });
}


// =============================================================================
// Marble Texture Effect (CSS-based enhancement)
// =============================================================================

function addMarbleTexture() {
    // Create subtle marble vein SVG pattern
    const marblePattern = `
        <svg xmlns="http://www.w3.org/2000/svg" width="400" height="400" viewBox="0 0 400 400">
            <defs>
                <filter id="noise" x="0%" y="0%" width="100%" height="100%">
                    <feTurbulence type="fractalNoise" baseFrequency="0.02" numOctaves="3" result="noise"/>
                    <feDiffuseLighting in="noise" lighting-color="#e8e4dc" surfaceScale="1">
                        <feDistantLight azimuth="45" elevation="60"/>
                    </feDiffuseLighting>
                </filter>
            </defs>
            <rect width="100%" height="100%" filter="url(#noise)" opacity="0.3"/>
        </svg>
    `;

    // Could be used for additional texture overlays
}


// =============================================================================
// Initialize Everything
// =============================================================================

document.addEventListener('DOMContentLoaded', () => {
    // Initialize particle background
    new ParticleBackground('bg-canvas');

    // Initialize GSAP animations
    initAnimations();

    // Initialize navigation
    initNavigation();

    // Add marble texture
    addMarbleTexture();

    // Log for debugging
    console.log('Matt Duffy website initialized');
});


// =============================================================================
// Performance: Reduce animations when not visible
// =============================================================================

document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        // Could pause heavy animations here
    } else {
        // Resume animations
    }
});
