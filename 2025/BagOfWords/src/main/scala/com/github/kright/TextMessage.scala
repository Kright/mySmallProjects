package com.github.kright

import java.io.File
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import Utils.MyFileExt
import ujson.Value

case class TextMessage(id: Int,
                       date: String,
                       from: String,
                       fromId: String,
                       replyToMessageId: Option[Int],
                       text: String)

object TextMessage{
  def fromFile(path: String): ArrayBuffer[TextMessage] = {
    val jsonText = new File(path).text
    val data = ujson.read(jsonText)
    val msgsObjs = data.obj("messages").arr.map(o => o.obj)
    val msgs = msgsObjs.flatMap(obj => TextMessage.fromMap(obj))
    msgs
  }

  private def fromMap(obj: mutable.Map[String, Value]): Option[TextMessage] = {
    if (!obj("type").strOpt.contains("message")) return None

    for(
      id <- obj("id").numOpt.map(_.toInt);
      date <- obj("date").strOpt;
      from <- obj("from").strOpt;
      fromId <- obj("from_id").strOpt;
      replyToMessageId = obj.get("reply_to_message_id").map(_.num.toInt);
      text <- obj("text").strOpt
    ) yield TextMessage(id, date, from, fromId, replyToMessageId, text)
  }
}
