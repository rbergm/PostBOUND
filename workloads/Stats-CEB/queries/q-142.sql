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
  AND c.CreationDate >= CAST('2010-07-26 17:09:48' AS timestamp)
  AND p.PostTypeId = 1
  AND p.AnswerCount >= 0
  AND p.CommentCount >= 0
  AND p.CommentCount <= 14
  AND pl.CreationDate >= CAST('2010-10-27 10:02:57' AS timestamp)
  AND pl.CreationDate <= CAST('2014-09-04 17:23:50' AS timestamp)
  AND ph.CreationDate <= CAST('2014-09-11 20:09:41' AS timestamp)
  AND v.CreationDate >= CAST('2010-07-21 00:00:00' AS timestamp)
  AND v.CreationDate <= CAST('2014-09-14 00:00:00' AS timestamp);
