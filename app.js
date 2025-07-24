document.addEventListener("DOMContentLoaded", function () {
    const loginSection = document.getElementById('login-section');
    const signupSection = document.getElementById('signup-section');
    const showSignupBtn = document.getElementById('show-signup');
    const showLoginBtn = document.getElementById('show-login');

    // Switch to Signup Form
    showSignupBtn.addEventListener('click', function () {
        loginSection.style.display = 'none';
        signupSection.style.display = 'block';
    });

    // Switch to Login Form
    showLoginBtn.addEventListener('click', function () {
        signupSection.style.display = 'none';
        loginSection.style.display = 'block';
    });

    // Login Form Submission
    document.getElementById('login-form').addEventListener('submit', async function (event) {
        event.preventDefault();
        const email = document.getElementById('login-email').value;
        const password = document.getElementById('login-password').value;
    
        const response = await fetch('/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password })
        });
    
        const result = await response.json();
        if (result.success) {
            // ✅ Redirect strictly based on role
            if (result.is_admin === true) {
                window.location.href = '/admin_dashboard';
            } else {
                window.location.href = '/chatbot';
            }
        } else {
            alert(result.error);
        }
    });

    
    // Signup Form Submission
    document.getElementById('signup-form').addEventListener('submit', async function (event) {
        event.preventDefault();
        const email = document.getElementById('signup-email').value;
        const password = document.getElementById('signup-password').value;

        const response = await fetch('/signup', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password })
        });

        const result = await response.json();
        if (result.success) {
            alert("Signup successful! You can now log in.");
            signupSection.style.display = 'none';
            loginSection.style.display = 'block';
        } else {
            alert(result.error);
        }
    });



});
