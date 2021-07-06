import random
from rest_framework.views import APIView
from rest_framework.authentication import SessionAuthentication, BasicAuthentication
from datetime import datetime
from rest_framework.response import Response
import os
from django.http import QueryDict, FileResponse
from main import init, process


class CsrfExemptSessionAuthentication(SessionAuthentication):

    def enforce_csrf(self, request):
        return


def fun(audio_file):
    return "a", "b", "14424.wav"


word2index, index2word, model, speaker = init()


class ReplyView(APIView):
    authentication_classes = (CsrfExemptSessionAuthentication, BasicAuthentication)

    def post(self, request):

        context = dict()
        context['err_code'] = 0
        file = request.FILES.get("file")  # 获取上传的文件，如果没有文件，则默认为None
        if not file:
            context['err_code'] = 2001
            context['error'] = "没有文件"
            return Response(context)

        dir_path = os.path.join(os.getcwd(), "media")
        file_name = str(datetime.now().year) + str(datetime.now().month) + str(datetime.now().day) + \
                    str(datetime.now().hour) + str(datetime.now().minute) + str(datetime.now().second) + \
                    str(random.randint(1, 1000)) + os.path.splitext(file.name)[1]

        while os.path.exists(file_name):
            file_name = str(datetime.now().year) + str(datetime.now().month) + str(datetime.now().day) + \
                        str(datetime.now().hour) + str(datetime.now().minute) + str(datetime.now().second) + \
                        str(random.randint(1, 1000)) + os.path.splitext(file.name)[1]
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        audio_file = os.path.join(dir_path, file_name)
        destination = open(audio_file, 'wb+')
        for chunk in file.chunks():
            destination.write(chunk)
        destination.close()

        submit_words, return_words, return_file = process(word2index, index2word, model, speaker, audio_file)

        context['submit_words'] = submit_words
        context['return_words'] = return_words
        context['return_files'] = return_file
        return Response(context)

    def get(self, request):
        file_name = request.GET.get('file')
        if file_name is None:
            context = dict()
            context['err_code'] = 3003
            context['error'] = "参数不正确"
            return Response(context)
        file = os.path.join(file_name)
        if not os.path.exists(file):
            context = dict()
            context['err_code'] = 4004
            context['error'] = "请求的文件不存在"
            return Response(context)
        else:
            file = open(file, 'rb')
            response = FileResponse(file)
            response['Content-Type'] = 'application/octet-stream'
            response['Content-Disposition'] = "attachment; filename= {}".format(file_name)
            return response
    # def get(self, request):
    #     context = dict()
    #     context['err_code'] = 0
    #     data = request.GET
    #     user = request.user
    #     if user.is_anonymous:
    #         context['err_code'] = 1001
    #         context['error'] = "您还未登录"
    #         return Response(context)
    #     work_id = data.get('work_id')
    #     print(work_id)
    #     try:
    #         work_id = int(work_id)
    #     except:
    #         context['err_code'] = 1002
    #         context['error'] = "请求参数不正确"
    #         return Response(context)
    #     done = HomeWorkMembersModel.objects.filter(work__id=work_id, owner=user)
    #     if not done.exists():
    #         context['err_code'] = 4004
    #         context['error'] = "您没有次参加本作业"
    #         return Response(context)
    #     done = done.first()
    #     context['data'] = dict()
    #     context['data']['done'] = done.done
    #     context['data']['file_name'] = done.file_name
    #     context['data']['work_name'] = done.work.name
    #     context['data']['upload_time'] = done.upload_time
    #     return Response(context)
