$(document).ready(function() {
    $(".toggle-rst-container > *").hide();
    $(".toggle-rst-container .header").show();

    $(".toggle-rst-container .header").click(function() {
        $(this).parent().children().not(".header").toggle(400);
        $(this).parent().children(".header").toggleClass("open");
    });
});