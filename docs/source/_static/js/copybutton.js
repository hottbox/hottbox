$(document).ready(function() {

    // Tooltip
    $('button').tooltip({
        trigger: 'click',
        placement: 'left'
    });

    function setTooltip(btn, message) {
        $(btn).tooltip('hide')
            .attr('data-original-title', message)
            .tooltip('show');
    }

    function hideTooltip(btn) {
        setTimeout(function () {
            $(btn).tooltip('hide');
        }, 1000);
    }

    function hidePrompt(btn){
        $(btn).parent().find('.go, .gp, .gt').hide();
    }

    function showPrompt(btn){
        $(btn).parent().find('.go, .gp, .gt').show();
    }

    function addCopyButtonToCode() {
        // get all code elements
        var allCodeBlocksElements = $("div.highlight pre");

        // For each element, do the following steps
        allCodeBlocksElements.each(function (ii) {
            // define a unique id for this element and add it
            var currentId = "codeblock" + (ii + 1);
            $(this).attr('id', currentId);

            // create a button that's configured for clipboard.js
            // point it to the text that's in this code block
            // add the button just after the text in the code block w/ jquery
            var clipButton = '<button class="btn copy-to-clipboard-btn" data-clipboard-target="#' + currentId + '"><img src="https://clipboardjs.com/assets/images/clippy.svg" width="15" alt="Copy to clipboard"></button>';
            $(this).after(clipButton);
        });

        // tell clipboard.js to look for clicks that match this query
        var clipboard = new ClipboardJS('.copy-to-clipboard-btn', {
            text: function(trigger) {
                // Copy only code snippet without prompts or outputs
                hidePrompt(trigger);
                return $(trigger).next('pre').find('.gt').content
            }
        });

        clipboard.on('success', function(e) {
            console.info('Action:', e.action);
            console.info('Text:', e.text);
            console.info('Trigger:', e.trigger);

            setTooltip(e.trigger, 'Copied!');
            e.clearSelection();
            showPrompt(e.trigger);
            hideTooltip(e.trigger);
        });

        clipboard.on('error', function(e) {
            console.error('Action:', e.action);
            console.error('Trigger:', e.trigger);
            setTooltip(e.trigger, 'Failed!');
            showPrompt(e.trigger);
            hideTooltip(e.trigger);
        });
    }

    addCopyButtonToCode();
});
