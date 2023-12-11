function chooseFile() {
    document.getElementById('id_image').click();
}
document.getElementById('myform').addEventListener('submit', function() {
    // submit 버튼을 클릭하면 파일 선택 input이 자동으로 실행됨
    document.getElementById('id_image').click();
});