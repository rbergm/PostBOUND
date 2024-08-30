SELECT COUNT(*)
FROM comments AS c,
  posts AS p,
  postLinks AS pl,
  postHistory AS ph,
  votes AS v,
  users AS u
WHERE u.Id = p.OwnerUserId
  AND p.Id = v.PostId
  AND p.Id = c.PostId
  AND p.Id = pl.PostId
  AND p.Id = ph.PostId
  AND p.PostTypeId = 1
  AND p.AnswerCount >= 0
  AND p.CreationDate >= CAST('2010-07-21 15:23:53' AS timestamp)
  AND p.CreationDate <= CAST('2014-09-11 23:26:14' AS timestamp)
  AND pl.CreationDate >= CAST('2010-11-16 01:27:37' AS timestamp)
  AND pl.CreationDate <= CAST('2014-08-21 15:25:23' AS timestamp)
  AND ph.PostHistoryTypeId = 5
  AND v.CreationDate >= CAST('2010-07-21 00:00:00' AS timestamp)
  AND u.UpVotes >= 0
  AND u.CreationDate <= CAST('2014-09-11 20:31:48' AS timestamp);
