

function load_main_layout(main){
    function _removeClasses(elements, class2remove) {
        for (var i = 0; i < elements.length; i++) {
            elements[i].classList.remove(class2remove);
        }
    }
    _removeClasses(document.querySelectorAll('.nuclei-io-navbar'), 'active');
    
    $('.nuclei-io-navbar#' + main).addClass("active");

    // hide other layout, show present layout
    $( "#mainlayout").find('.layout_child').hide()

    $( "#mainlayout #layout_"+main ).show();
}
