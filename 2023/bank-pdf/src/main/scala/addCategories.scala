import java.io.File
import scala.collection.mutable.ArrayBuffer

@main
def groupData(): Unit = {
  val root = new File("/home/lgor/projects/2024/files/2024/fin/csv")

  val categories = Categories.parseAll(new File(root, "categories.csv"))

  val files = root.listFiles().filter { file =>
    val name = file.getName
    name.matches("""\d{4}[-]\d{2}.csv""")
  }
  println(s"files = ${files.mkString("[", ", ", "]")}")

  for (f <- files) {
    val (csvNames, csv) = f.readCsv()

    println(s"csv = ${csv}")

    val csvWithCategories = csv.map { line =>
      line + ("category" -> categories.getCategory(line("opis")).map(_.name).getOrElse("UNKNOWN"))
    }

    new File(f.getParent, f.getName - ".csv" + "-cat.csv").writeCsv(csvNames :+ "category", csvWithCategories.sortBy(_("category")))

    val groupedByCategories = groupByCategories(csvWithCategories)

    new File(f.getParent, f.getName - ".csv" + "-by-cat.csv")
      .writeCsv(Seq("category", "in", "out"), Seq(makeSummary(groupedByCategories)) ++ groupedByCategories)
  }

  val totalGroupedByCategories = groupByCategories(files.filter(_.getName.startsWith("2023")).flatMap { f =>
    val (csvNames, csv) = f.readCsv()
    val csvWithCategories = csv.map { line =>
      line + ("category" -> categories.getCategory(line("opis")).map(_.name).getOrElse("UNKNOWN"))
    }
    csvWithCategories
  })

  new File(files.head.getParent, "2023-by-cat.csv")
    .writeCsv(Seq("category", "in", "out"), (Seq(makeSummary(totalGroupedByCategories)) ++ totalGroupedByCategories).sortBy(- _("out").toInt))
}

private def makeSummary(groupedByCategories: Iterable[Map[String, String]]): Map[String, String] = Map(
  "category" -> "ВСЕГО",
  "in" -> groupedByCategories.map(_("in").toInt).sum.toString,
  "out" -> groupedByCategories.map(_("out").toInt).sum.toString,
)

private def groupByCategories(csvWithCategories: Seq[Map[String, String]]): Iterable[Map[String, String]] =
  csvWithCategories.groupBy(m => m("category")).flatMap { (category, lines) =>
    val isplata = lines.map(_("isplata")).filterNot(_.isBlank).map(_.split(" ").head.toDouble.toInt).sum
    val uplata = lines.map(_("uplata")).filterNot(_.isBlank).map(_.split(" ").head.toDouble.toInt).sum

    if (isplata != 0.0 || uplata != 0) {
      Option(Map("category" -> category, "in" -> uplata.toString, "out" -> isplata.toString))
    }
    else None
  }


case class Category(name: String, pattern: String)


class Categories(val elems: Seq[Category]):
  def getCategory(name: String): Option[Category] =
    val matched = elems.filter(e => name.contains(e.pattern))
    if (matched.size > 1) {
      val name = matched.head.name
      require(matched.forall(_.name == name), s"many matches for '$name': ${matched}")
    }
    matched.headOption

object Categories:
  def parseAll(file: File): Categories =
    val (csvNames, csv) = file.readCsv()
    new Categories(csv.map { m =>
      Category(m("category name"), m("pattern"))
    })