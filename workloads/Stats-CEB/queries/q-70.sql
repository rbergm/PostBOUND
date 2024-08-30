SELECT COUNT(*)
FROM comments AS c,
  posts AS p,
  postLinks AS pl,
  postHistory AS ph,
  votes AS v,
  badges AS b
WHERE p.Id = c.PostId
  AND p.Id = pl.RelatedPostId
  AND p.Id = ph.PostId
  AND p.Id = v.PostId
  AND b.UserId = c.UserId
  AND c.CreationDate >= CAST('2010-07-26 20:21:15' AS timestamp)
  AND c.CreationDate <= CAST('2014-09-13 18:12:10' AS timestamp)
  AND p.Score <= 61
  AND p.ViewCount <= 3627
  AND p.AnswerCount >= 0
  AND p.AnswerCount <= 5
  AND p.CommentCount <= 8
  AND p.FavoriteCount >= 0
  AND v.VoteTypeId = 2
  AND v.CreationDate >= CAST('2010-07-27 00:00:00' AS timestamp)
  AND b.Date >= CAST('2010-07-30 03:49:24' AS timestamp);
