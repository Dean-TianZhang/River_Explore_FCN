window.addEvent('domready',function(){
	var form='page-contact';
	new FormCheck($(form),{
		submitByAjax:true,
		onValidateSuccess:function(){
			$('contact_send').setStyle('cursor','default');
			$('contact_send').setStyle('background','#aaa');
		},
		onAjaxSuccess:function(){
			$$('#'+form+' input[type="text"]').append($$('#'+form+' textarea')).each(function(el) {el.set('value','');});
			$$('#'+form+' input[type="radio"]').removeProperty('checked');
			$('contact_send').setStyle('display','inline');		
			$(form).getElement('span.no_error').morph({'opacity':1,'height':10,'padding-top':12,'padding-bottom':22,'margin-bottom':20});
			(function(){$(form).getElement('span.no_error').morph({'opacity':0,'height':0,'padding':0,'margin-bottom':0});}).delay(7000);
		},
		display:{
			showErrors:1,
			titlesInsteadNames:true,
			scrollToFirst:false,
			indicateErrors:2
		}
	});
});
