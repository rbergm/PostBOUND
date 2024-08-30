SELECT COUNT(*)
FROM comments AS c,
  posts AS p,
  postLinks AS pl,
  postHistory AS ph,
  votes AS v,
  badges AS b,
  users AS u
WHERE p.Id = pl.RelatedPostId
  AND b.UserId = u.Id
  AND c.UserId = u.Id
  AND p.Id = v.PostId
  AND p.Id = c.PostId
  AND p.Id = ph.PostId
  AND c.Score = 0
  AND p.ViewCount >= 0
  AND p.AnswerCount <= 5
  AND p.CommentCount <= 12
  AND p.FavoriteCount >= 0
  AND pl.LinkTypeId = 1
  AND pl.CreationDate >= CAST('2011-02-16 20:04:50' AS timestamp)
  AND pl.CreationDate <= CAST('2014-09-01 16:48:04' AS timestamp)
  AND v.CreationDate >= CAST('2010-07-19 00:00:00' AS timestamp)
  AND v.CreationDate <= CAST('2014-08-31 00:00:00' AS timestamp)
  AND b.Date >= CAST('2010-08-06 10:36:45' AS timestamp)
  AND b.Date <= CAST('2014-09-12 07:19:35' AS timestamp);
