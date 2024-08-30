SELECT COUNT(*)
FROM comments AS c,
  posts AS p,
  postLinks AS pl,
  postHistory AS ph,
  votes AS v
WHERE p.Id = pl.PostId
  AND p.Id = v.PostId
  AND p.Id = ph.PostId
  AND p.Id = c.PostId
  AND c.CreationDate <= CAST('2014-09-10 02:42:35' AS timestamp)
  AND p.Score >= -1
  AND p.ViewCount <= 5896
  AND p.AnswerCount >= 0
  AND p.CreationDate >= CAST('2010-07-29 15:57:21' AS timestamp)
  AND v.VoteTypeId = 2;
