(function() {
	function computeBMI(h, w) {
		var hm = Number(h || 0) / 100;
		var kg = Number(w || 0);
		if (!hm || hm <= 0 || !kg) return '';
		return (kg / (hm * hm)).toFixed(1);
	}
	var height = document.getElementById('height');
	var weight = document.getElementById('weight');
	var bmi = document.getElementById('bmi');
	function updateBMI() {
		if (!bmi) return;
		bmi.value = computeBMI(height && height.value, weight && weight.value) || '';
	}
	if (height) height.addEventListener('input', updateBMI);
	if (weight) weight.addEventListener('input', updateBMI);
	updateBMI();
})();
